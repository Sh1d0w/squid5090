// TigerStyle Inference Engine: single-arena runtime VRAM, 48-visit Blackwell forward pass,
// CUDA Graphs (B=1 decode, speculative B=2..8, prefill B=8), aux_stream DSpark overlap,
// zero-allocation streaming token generation, and long-context benchmark.

const std = @import("std");
const constants = @import("constants.zig");
const cuda = @import("cuda.zig");
const tiger = @import("tiger.zig");
const vram_arena = @import("vram_arena.zig");
const weights_mod = @import("weights.zig");

pub const TokenCallbackFn = *const fn (context: *anyopaque, token_id: i32) bool;

pub const GenerateStats = struct {
    prompt_tokens: u32 = 0,
    generated_tokens: u32 = 0,
    prefill_ms: f64 = 0.0,
    decode_ms: f64 = 0.0,
    spec_steps: u32 = 0,
    total_drafts: u32 = 0,
    total_accepted: u32 = 0,
    last_token_id: i32 = -1,
};

pub const Engine = struct {
    weights: ?*const weights_mod.ModelWeights = null,
    context_tokens_max: u32 = 0,
    use_graphs: bool = true,
    kv_format: constants.KVFormat = .fp8,
    current_position: i32 = 0,
    schedule: [constants.visit_count]constants.VisitInfo = undefined,

    arena: vram_arena.VRAMArena = .{},
    stream: ?*anyopaque = null,
    aux_stream: ?*anyopaque = null,
    fork_event: ?*anyopaque = null,
    join_event: ?*anyopaque = null,

    // Cached CUDA Graphs
    b1_graph_exec: ?*anyopaque = null,
    spec_graph_exec: [constants.mtp_draft_count_max + 1]?*anyopaque = @splat(null),
    prefill_b8_graph_exec: [2]?*anyopaque = .{ null, null },

    // Pinned host readback buffers (zero-malloc decode loop)
    h_accepted: ?[*]i32 = null,
    h_num_accepted: ?*i32 = null,

    // Per-visit KV caches and sconv rings
    kv_cap: [constants.visit_count]i32 = @splat(0),
    k_cache: [constants.visit_count]?*anyopaque = @splat(null),
    v_cache: [constants.visit_count]?*anyopaque = @splat(null),
    k_scale: [constants.visit_count]?*anyopaque = @splat(null),
    v_scale: [constants.visit_count]?*anyopaque = @splat(null),
    k_ring: [constants.visit_count][*]f32 = undefined,
    v_ring: [constants.visit_count][*]f32 = undefined,
    attn_ring: [constants.visit_count][*]f32 = undefined,
    mlp_ring: [constants.visit_count][*]f32 = undefined,

    // DSpark KV caches and f_ring
    ds_kv_cap: [constants.mtp_layer_count]i32 = .{ 0, 0, 0 },
    ds_k_cache: [constants.mtp_layer_count]?*anyopaque = .{ null, null, null },
    ds_v_cache: [constants.mtp_layer_count]?*anyopaque = .{ null, null, null },
    ds_k_scale: [constants.mtp_layer_count]?*anyopaque = .{ null, null, null },
    ds_v_scale: [constants.mtp_layer_count]?*anyopaque = .{ null, null, null },
    ds_f_ring: ?[*]f32 = null,

    // Activation scratch arena (sized for B = mtp_block_size_max = 8)
    split_buf: [*]f32 = undefined,
    d_pos: *i32 = undefined,
    d_tokens: [*]i32 = undefined,
    d_next_tokens: [*]i32 = undefined,
    d_anchor: *i32 = undefined,
    d_drafts: [*]i32 = undefined,
    d_accepted: [*]i32 = undefined,
    d_num_acc: *i32 = undefined,

    hidden: [*]f32 = undefined,
    prelude: [*]f32 = undefined,
    partial: [*]f32 = undefined,
    blocks: [*]f32 = undefined,
    saved: [*]f32 = undefined,
    prior_last: [*]f32 = undefined,
    attn_in: [3][*]f32 = undefined,
    attn_in_h: [3]*anyopaque = undefined,

    qrgs: [*]f32 = undefined,
    k_proj: [*]f32 = undefined,
    v_proj: [*]f32 = undefined,
    attn_gated: [*]f32 = undefined,
    attn_gated_h: *anyopaque = undefined,
    o_proj: [*]f32 = undefined,
    residual: [*]f32 = undefined,
    mlp_in: [*]f32 = undefined,
    mlp_normed: [*]f32 = undefined,
    mlp_normed_h: *anyopaque = undefined,
    ctrl_msg: [*]f32 = undefined,
    ctrl_cat: [*]f32 = undefined,
    ctrl_d: [*]f32 = undefined,
    router_logits: [*]f32 = undefined,
    route_ids: [*]i32 = undefined,
    route_w: [*]f32 = undefined,
    exp_groups: *anyopaque = undefined,
    moe_act_h: *anyopaque = undefined,
    moe_down: [*]f32 = undefined,
    moe_slot_out: [*]f32 = undefined,
    taps: [*]f32 = undefined,
    ds_mix: [*]f32 = undefined,
    ds_kv: [*]f32 = undefined,
    ds_ws: [*]f32 = undefined,
    lm_blk_val: [*]f32 = undefined,
    lm_blk_idx: [*]i32 = undefined,

    pub fn init(
        self: *Engine,
        weights: *const weights_mod.ModelWeights,
        context_tokens_max: u32,
        use_graphs: bool,
        kv_format: constants.KVFormat,
    ) void {
        tiger.assert(context_tokens_max >= 64);
        tiger.assert(self.arena.base_ptr == null);

        self.weights = weights;
        self.context_tokens_max = context_tokens_max;
        self.use_graphs = use_graphs;
        self.kv_format = kv_format;
        self.schedule = constants.build_visit_schedule();

        tiger.assert_cuda(cuda.squid_cuda_stream_create(&self.stream));
        tiger.assert_cuda(cuda.squid_cuda_stream_create(&self.aux_stream));
        tiger.assert_cuda(cuda.squid_cuda_event_create(&self.fork_event, 1));
        tiger.assert_cuda(cuda.squid_cuda_event_create(&self.join_event, 1));

        const expert_groups_bytes: u64 = cuda.squid_sizeof_expert_groups();
        tiger.assert(expert_groups_bytes > 0);
        const has_dspark: bool = (weights.dspark.loaded != 0);

        // Pass 1: compute exact aligned runtime VRAM requirement -> 1 single cudaMalloc!
        const runtime_bytes: u64 = compute_runtime_vram_bytes(
            &self.schedule,
            context_tokens_max,
            kv_format,
            has_dspark,
            expert_groups_bytes,
        );
        self.arena = vram_arena.VRAMArena.init(runtime_bytes);

        // Pass 2: carve every runtime buffer from the single contiguous VRAMArena
        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            const is_glob: bool = self.schedule[vi].is_global;
            const cap: u32 = if (is_glob) context_tokens_max else (constants.swa_window_tokens + constants.mtp_block_size_max);
            const vfmt: constants.KVFormat = if (is_glob) kv_format else .bf16;
            self.kv_cap[vi] = @intCast(cap);
            self.alloc_kv_pair(cap, vfmt, &self.k_cache[vi], &self.v_cache[vi], &self.k_scale[vi], &self.v_scale[vi]);
            self.k_ring[vi] = self.arena.alloc_many(f32, constants.sconv_ring_slots * constants.kv_dim);
            self.v_ring[vi] = self.arena.alloc_many(f32, constants.sconv_ring_slots * constants.kv_dim);
            self.attn_ring[vi] = self.arena.alloc_many(f32, constants.sconv_ring_slots * constants.hidden_dim);
            self.mlp_ring[vi] = self.arena.alloc_many(f32, constants.sconv_ring_slots * constants.hidden_dim);
        }

        if (has_dspark) {
            var ml: u32 = 0;
            while (ml < constants.mtp_layer_count) : (ml += 1) {
                const is_loc: bool = constants.is_mtp_local_layer(ml);
                const cap: u32 = if (is_loc)
                    (constants.swa_window_tokens + constants.mtp_block_size_max)
                else if (context_tokens_max > 524_288)
                    65_536
                else
                    context_tokens_max;
                const lfmt: constants.KVFormat = if (is_loc) .bf16 else kv_format;
                self.ds_kv_cap[ml] = @intCast(cap);
                self.alloc_kv_pair(cap, lfmt, &self.ds_k_cache[ml], &self.ds_v_cache[ml], &self.ds_k_scale[ml], &self.ds_v_scale[ml]);
            }
            self.ds_f_ring = self.arena.alloc_many(f32, constants.sconv_ring_slots * constants.hidden_dim);
        }

        self.split_buf = self.arena.alloc_many(f32, constants.split_buffer_floats);

        const b_max: u64 = constants.mtp_block_size_max;
        self.d_pos = self.arena.alloc_one(i32);
        self.d_tokens = self.arena.alloc_many(i32, b_max);
        self.d_next_tokens = self.arena.alloc_many(i32, b_max);
        self.d_anchor = self.arena.alloc_one(i32);
        self.d_drafts = self.arena.alloc_many(i32, constants.mtp_draft_count_max);
        self.d_accepted = self.arena.alloc_many(i32, b_max);
        self.d_num_acc = self.arena.alloc_one(i32);

        var h_acc_raw: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&h_acc_raw, @intCast(b_max * @sizeOf(i32)), 0));
        self.h_accepted = @ptrCast(@alignCast(h_acc_raw.?));

        var h_nacc_raw: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_malloc_host_pinned(&h_nacc_raw, @sizeOf(i32), 0));
        self.h_num_accepted = @ptrCast(@alignCast(h_nacc_raw.?));

        self.hidden = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.prelude = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.partial = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.blocks = self.arena.alloc_many(f32, b_max * 4 * constants.hidden_dim);
        self.saved = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.prior_last = self.arena.alloc_many(f32, b_max * constants.hidden_dim);

        var s_idx: u32 = 0;
        while (s_idx < 3) : (s_idx += 1) {
            self.attn_in[s_idx] = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
            self.attn_in_h[s_idx] = self.arena.alloc_raw(b_max * constants.hidden_dim * 2);
        }

        self.qrgs = self.arena.alloc_many(f32, b_max * constants.qrgs_out_dim);
        self.k_proj = self.arena.alloc_many(f32, b_max * constants.kv_dim);
        self.v_proj = self.arena.alloc_many(f32, b_max * constants.kv_dim);
        self.attn_gated = self.arena.alloc_many(f32, b_max * constants.query_dim);
        self.attn_gated_h = self.arena.alloc_raw(b_max * constants.query_dim * 2);
        self.o_proj = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.residual = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.mlp_in = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.mlp_normed = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.mlp_normed_h = self.arena.alloc_raw(b_max * constants.hidden_dim * 2);
        self.ctrl_msg = self.arena.alloc_many(f32, 3 * b_max * constants.hidden_dim);
        self.ctrl_cat = self.arena.alloc_many(f32, 3 * b_max * 2 * constants.hidden_dim);
        self.ctrl_d = self.arena.alloc_many(f32, 3 * b_max * constants.gate_rank);
        self.router_logits = self.arena.alloc_many(f32, b_max * constants.expert_count);
        self.route_ids = self.arena.alloc_many(i32, b_max * constants.top_k_experts);
        self.route_w = self.arena.alloc_many(f32, b_max * constants.top_k_experts);
        self.exp_groups = self.arena.alloc_raw(expert_groups_bytes);
        self.moe_act_h = self.arena.alloc_raw(b_max * constants.top_k_experts * constants.intermediate_dim * 2);
        self.moe_down = self.arena.alloc_many(f32, b_max * constants.hidden_dim);
        self.moe_slot_out = self.arena.alloc_many(f32, b_max * constants.top_k_experts * constants.hidden_dim);
        self.taps = self.arena.alloc_many(f32, 4 * b_max * constants.hidden_dim);
        self.ds_mix = self.arena.alloc_many(f32, 3 * b_max * constants.hidden_dim);
        self.ds_kv = self.arena.alloc_many(f32, b_max * 2 * constants.kv_dim);
        self.ds_ws = self.arena.alloc_many(f32, constants.dspark_workspace_floats);
        self.lm_blk_val = self.arena.alloc_many(f32, b_max * 1024);
        self.lm_blk_idx = self.arena.alloc_many(i32, b_max * 1024);

        // Paired TigerStyle invariant: Pass 1 == Pass 2!
        self.arena.assert_exact_fill();

        const rt_gib: f64 = @as(f64, @floatFromInt(self.arena.capacity_bytes)) / (1024.0 * 1024.0 * 1024.0);
        tiger.print_stdout(
            "[engine] Initialized max_ctx={d} | kv={s} | runtime VRAM={d:.2} GiB (1 arena, {d} slices) | graphs={s}\n",
            .{ context_tokens_max, kv_format.name(), rt_gib, self.arena.allocation_count, if (use_graphs) "on" else "off" },
        );
        self.reset();
    }

    pub fn deinit(self: *Engine) void {
        if (self.b1_graph_exec) |g| tiger.assert_cuda(cuda.squid_cuda_graph_destroy(g));
        for (&self.spec_graph_exec) |*g| {
            if (g.*) |exec| tiger.assert_cuda(cuda.squid_cuda_graph_destroy(exec));
            g.* = null;
        }
        for (&self.prefill_b8_graph_exec) |*g| {
            if (g.*) |exec| tiger.assert_cuda(cuda.squid_cuda_graph_destroy(exec));
            g.* = null;
        }
        if (self.h_accepted) |p| tiger.assert_cuda(cuda.squid_cuda_free_host(@ptrCast(p)));
        if (self.h_num_accepted) |p| tiger.assert_cuda(cuda.squid_cuda_free_host(@ptrCast(p)));
        self.arena.deinit();
        if (self.fork_event) |e| tiger.assert_cuda(cuda.squid_cuda_event_destroy(e));
        if (self.join_event) |e| tiger.assert_cuda(cuda.squid_cuda_event_destroy(e));
        if (self.aux_stream) |s| tiger.assert_cuda(cuda.squid_cuda_stream_destroy(s));
        if (self.stream) |s| tiger.assert_cuda(cuda.squid_cuda_stream_destroy(s));
        self.* = .{};
    }

    pub fn reset(self: *Engine) void {
        tiger.assert(self.stream != null);
        self.current_position = 0;
        tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(self.d_pos), 0, @sizeOf(i32), self.stream));
        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(self.k_ring[vi]), 0, constants.sconv_ring_slots * constants.kv_dim * 4, self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(self.v_ring[vi]), 0, constants.sconv_ring_slots * constants.kv_dim * 4, self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(self.attn_ring[vi]), 0, constants.sconv_ring_slots * constants.hidden_dim * 4, self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(self.mlp_ring[vi]), 0, constants.sconv_ring_slots * constants.hidden_dim * 4, self.stream));
        }
        if (self.ds_f_ring) |fr| {
            tiger.assert_cuda(cuda.squid_cuda_memset_async(@ptrCast(fr), 0, constants.sconv_ring_slots * constants.hidden_dim * 4, self.stream));
        }
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
    }

    /// Generate up to `max_new_tokens` into `out_tokens` (and optionally invoke `on_token`).
    /// ZERO heap allocations occur during the entire prefill + decode execution!
    pub fn generate(
        self: *Engine,
        prompt: []const i32,
        max_new_tokens_in: u32,
        num_drafts_in: u32,
        out_tokens: []i32,
        on_token: ?TokenCallbackFn,
        callback_context: ?*anyopaque,
    ) GenerateStats {
        tiger.assert(self.weights != null);
        var stats: GenerateStats = .{ .prompt_tokens = @intCast(prompt.len) };
        if (prompt.len == 0 or max_new_tokens_in == 0) return stats;
        if (prompt.len >= self.context_tokens_max) return stats;

        var max_new_tokens: u32 = max_new_tokens_in;
        const room: u32 = self.context_tokens_max - @as(u32, @intCast(prompt.len));
        if (max_new_tokens > room) max_new_tokens = room;
        if (out_tokens.len > 0 and max_new_tokens > out_tokens.len) {
            max_new_tokens = @intCast(out_tokens.len);
        }

        const w = self.weights.?;
        const num_drafts: u32 = if (num_drafts_in > constants.mtp_draft_count_max) constants.mtp_draft_count_max else num_drafts_in;
        const use_mtp: bool = (num_drafts > 0 and w.dspark.loaded != 0);

        self.reset();

        // --- PREFILL ---
        const prompt_len: u32 = @intCast(prompt.len);
        if (self.use_graphs and prompt_len > constants.mtp_block_size_max) {
            self.ensure_prefill_b8_graph(use_mtp);
        }

        const prefill_start_ns: u64 = tiger.clock_monotonic_ns();
        var i: u32 = 0;
        if (self.use_graphs and prompt_len > constants.mtp_block_size_max) {
            const pgraph = self.prefill_b8_graph_exec[if (use_mtp) 1 else 0].?;
            while (i + constants.mtp_block_size_max < prompt_len) {
                tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(
                    @ptrCast(self.d_tokens),
                    @ptrCast(prompt[i..].ptr),
                    constants.mtp_block_size_max * @sizeOf(i32),
                    self.stream,
                ));
                tiger.assert_cuda(cuda.squid_cuda_graph_launch(pgraph, self.stream));
                self.current_position += constants.mtp_block_size_max;
                i += constants.mtp_block_size_max;
            }
        }
        while (i < prompt_len) {
            const batch: u32 = @intCast(tiger.min_u64(constants.mtp_block_size_max, prompt_len - i));
            const is_last: bool = (i + batch == prompt_len);
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_tokens), @ptrCast(prompt[i..].ptr), batch * @sizeOf(i32), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
            self.record_target_forward(@intCast(batch), use_mtp, is_last);
            self.current_position += @intCast(batch);
            i += batch;
        }

        const last_batch: u32 = ((prompt_len - 1) % constants.mtp_block_size_max) + 1;
        var first_tok: i32 = 0;
        tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(&first_tok), @ptrCast(self.d_next_tokens + (last_batch - 1)), @sizeOf(i32), self.stream));
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&self.current_position), @sizeOf(i32), self.stream));
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));
        stats.prefill_ms = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - prefill_start_ns)) / 1_000_000.0;

        if (out_tokens.len > 0) out_tokens[0] = first_tok;
        stats.generated_tokens = 1;
        stats.last_token_id = first_tok;
        if (on_token) |cb| {
            if (!cb(callback_context.?, first_tok)) return stats;
        }
        if (first_tok == constants.eos_token_id or stats.generated_tokens >= max_new_tokens) return stats;

        // Prepare decode CUDA Graph before starting decode timer
        if (!use_mtp) {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_tokens), @ptrCast(&first_tok), @sizeOf(i32), self.stream));
            self.ensure_b1_graph();
        } else {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_anchor), @ptrCast(&first_tok), @sizeOf(i32), self.stream));
            self.ensure_spec_graph(num_drafts);
        }
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

        // --- DECODE ---
        const decode_start_ns: u64 = tiger.clock_monotonic_ns();
        if (!use_mtp) {
            while (stats.generated_tokens < max_new_tokens) {
                if (self.use_graphs) {
                    tiger.assert_cuda(cuda.squid_cuda_graph_launch(self.b1_graph_exec.?, self.stream));
                } else {
                    self.record_target_forward(1, false, true);
                    cuda.squid_launch_inc_pos_and_copy_tok(self.d_pos, &self.d_tokens[0], &self.d_next_tokens[0], self.stream);
                }
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(self.h_accepted.?), @ptrCast(self.d_next_tokens), @sizeOf(i32), self.stream));
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

                const tok: i32 = self.h_accepted.?[0];
                if (out_tokens.len > 0) out_tokens[stats.generated_tokens] = tok;
                stats.generated_tokens += 1;
                stats.last_token_id = tok;
                self.current_position += 1;
                if (on_token) |cb| {
                    if (!cb(callback_context.?, tok)) break;
                }
                if (tok == constants.eos_token_id) break;
            }
        } else {
            const batch: i32 = @intCast(num_drafts + 1);
            while (stats.generated_tokens < max_new_tokens) {
                if (self.use_graphs) {
                    tiger.assert_cuda(cuda.squid_cuda_graph_launch(self.spec_graph_exec[num_drafts].?, self.stream));
                } else {
                    const ds_ks_const: [constants.mtp_layer_count]?*const anyopaque = .{ self.ds_k_scale[0], self.ds_k_scale[1], self.ds_k_scale[2] };
                    const ds_vs_const: [constants.mtp_layer_count]?*const anyopaque = .{ self.ds_v_scale[0], self.ds_v_scale[1], self.ds_v_scale[2] };
                    cuda.squid_launch_dspark_draft(
                        w.embed.?,
                        w.embed_norm.?,
                        &w.dspark,
                        self.d_anchor,
                        self.d_pos,
                        @intCast(num_drafts),
                        &self.ds_k_cache,
                        &self.ds_v_cache,
                        &ds_ks_const,
                        &ds_vs_const,
                        &self.ds_kv_cap,
                        @intFromEnum(self.kv_format),
                        self.split_buf,
                        self.ds_f_ring.?,
                        self.ds_ws,
                        self.d_drafts,
                        self.stream,
                    );
                    cuda.squid_launch_assemble_verify_tokens(self.d_anchor, self.d_drafts, self.d_tokens, @intCast(num_drafts), self.stream);
                    self.record_target_forward(batch, true, true);
                    cuda.squid_launch_spec_verify(self.d_next_tokens, self.d_drafts, batch, self.d_accepted, self.d_num_acc, self.d_pos, self.d_anchor, self.stream);
                }

                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(self.h_num_accepted.?), @ptrCast(self.d_num_acc), @sizeOf(i32), self.stream));
                tiger.assert_cuda(cuda.squid_cuda_memcpy_d2h_async(@ptrCast(self.h_accepted.?), @ptrCast(self.d_accepted), @as(usize, @intCast(batch)) * @sizeOf(i32), self.stream));
                tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

                const n_acc_i32: i32 = self.h_num_accepted.?.*;
                tiger.assert(n_acc_i32 >= 1 and n_acc_i32 <= batch);
                const n_acc: u32 = @intCast(n_acc_i32);

                stats.spec_steps += 1;
                stats.total_drafts += num_drafts;
                stats.total_accepted += (n_acc - 1);
                self.current_position += n_acc_i32;

                var stop_now: bool = false;
                var idx: u32 = 0;
                while (idx < n_acc and stats.generated_tokens < max_new_tokens) : (idx += 1) {
                    const tok: i32 = self.h_accepted.?[idx];
                    if (out_tokens.len > 0) out_tokens[stats.generated_tokens] = tok;
                    stats.generated_tokens += 1;
                    stats.last_token_id = tok;
                    if (on_token) |cb| {
                        if (!cb(callback_context.?, tok)) {
                            stop_now = true;
                            break;
                        }
                    }
                    if (tok == constants.eos_token_id) {
                        stop_now = true;
                        break;
                    }
                }
                if (stop_now) break;
            }
        }
        stats.decode_ms = @as(f64, @floatFromInt(tiger.clock_monotonic_ns() - decode_start_ns)) / 1_000_000.0;
        return stats;
    }

    /// Long-context latency microbenchmark at simulated KV position `sim_pos`.
    pub fn bench_step_at_pos(self: *Engine, sim_pos_in: i32, num_drafts: u32, iterations: u32) f64 {
        tiger.assert(iterations > 0 and iterations <= 256);
        const w = self.weights.?;
        const use_mtp: bool = (num_drafts > 0 and w.dspark.loaded != 0);
        const max_allowed: i32 = @as(i32, @intCast(self.context_tokens_max)) - @as(i32, @intCast(iterations * constants.mtp_block_size_max)) - 32;
        var sim_pos: i32 = sim_pos_in;
        if (sim_pos > max_allowed) sim_pos = tiger.max_i32(0, max_allowed);

        self.reset();

        // Zero-init global KV caches so synthetic reads are strictly finite (zero NaN)
        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            if (!self.schedule[vi].is_global) continue;
            const cap: u64 = @intCast(self.kv_cap[vi]);
            const data_bytes: usize = @intCast(switch (self.kv_format) {
                .bf16 => cap * constants.kv_dim * 2,
                .fp8 => cap * constants.kv_dim,
                .q4 => cap * (constants.kv_dim / 2),
            });
            tiger.assert_cuda(cuda.squid_cuda_memset_async(self.k_cache[vi].?, 0, data_bytes, self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memset_async(self.v_cache[vi].?, 0, data_bytes, self.stream));
            if (self.k_scale[vi]) |ks| {
                const scale_bytes: usize = @intCast(if (self.kv_format == .fp8)
                    cap * constants.kv_head_count * 2
                else
                    cap * constants.kv_q4_scales_per_token * 2);
                tiger.assert_cuda(cuda.squid_cuda_memset_async(ks, 0, scale_bytes, self.stream));
                tiger.assert_cuda(cuda.squid_cuda_memset_async(self.v_scale[vi].?, 0, scale_bytes, self.stream));
            }
        }
        if (use_mtp) {
            const cap: u64 = @intCast(self.ds_kv_cap[1]);
            const data_bytes: usize = @intCast(switch (self.kv_format) {
                .bf16 => cap * constants.kv_dim * 2,
                .fp8 => cap * constants.kv_dim,
                .q4 => cap * (constants.kv_dim / 2),
            });
            tiger.assert_cuda(cuda.squid_cuda_memset_async(self.ds_k_cache[1].?, 0, data_bytes, self.stream));
            tiger.assert_cuda(cuda.squid_cuda_memset_async(self.ds_v_cache[1].?, 0, data_bytes, self.stream));
            if (self.ds_k_scale[1]) |ks| {
                const scale_bytes: usize = @intCast(if (self.kv_format == .fp8)
                    cap * constants.kv_head_count * 2
                else
                    cap * constants.kv_q4_scales_per_token * 2);
                tiger.assert_cuda(cuda.squid_cuda_memset_async(ks, 0, scale_bytes, self.stream));
                tiger.assert_cuda(cuda.squid_cuda_memset_async(self.ds_v_scale[1].?, 0, scale_bytes, self.stream));
            }
        }

        const seed_tok: i32 = 150_000;
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_tokens), @ptrCast(&seed_tok), @sizeOf(i32), self.stream));
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_anchor), @ptrCast(&seed_tok), @sizeOf(i32), self.stream));
        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&sim_pos), @sizeOf(i32), self.stream));

        if (!use_mtp) self.ensure_b1_graph() else self.ensure_spec_graph(num_drafts);
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

        const graph_exec = if (use_mtp) self.spec_graph_exec[num_drafts].? else self.b1_graph_exec.?;
        var warmup: u32 = 0;
        while (warmup < 4) : (warmup += 1) {
            tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&sim_pos), @sizeOf(i32), self.stream));
            tiger.assert_cuda(cuda.squid_cuda_graph_launch(graph_exec, self.stream));
        }
        tiger.assert_cuda(cuda.squid_cuda_stream_sync(self.stream));

        tiger.assert_cuda(cuda.squid_cuda_memcpy_h2d_async(@ptrCast(self.d_pos), @ptrCast(&sim_pos), @sizeOf(i32), self.stream));
        var ev0: ?*anyopaque = null;
        var ev1: ?*anyopaque = null;
        tiger.assert_cuda(cuda.squid_cuda_event_create(&ev0, 0));
        tiger.assert_cuda(cuda.squid_cuda_event_create(&ev1, 0));
        defer tiger.assert_cuda(cuda.squid_cuda_event_destroy(ev0));
        defer tiger.assert_cuda(cuda.squid_cuda_event_destroy(ev1));

        tiger.assert_cuda(cuda.squid_cuda_event_record(ev0, self.stream));
        var it: u32 = 0;
        while (it < iterations) : (it += 1) {
            tiger.assert_cuda(cuda.squid_cuda_graph_launch(graph_exec, self.stream));
        }
        tiger.assert_cuda(cuda.squid_cuda_event_record(ev1, self.stream));
        tiger.assert_cuda(cuda.squid_cuda_event_sync(ev1));

        var elapsed_ms: f32 = 0.0;
        tiger.assert_cuda(cuda.squid_cuda_event_elapsed_ms(&elapsed_ms, ev0, ev1));
        return @as(f64, @floatCast(elapsed_ms)) / @as(f64, @floatFromInt(iterations));
    }

    fn alloc_kv_pair(
        self: *Engine,
        cap_tokens: u32,
        kv_format: constants.KVFormat,
        k_out: *?*anyopaque,
        v_out: *?*anyopaque,
        ks_out: *?*anyopaque,
        vs_out: *?*anyopaque,
    ) void {
        const cap: u64 = cap_tokens;
        switch (kv_format) {
            .bf16 => {
                const bytes = cap * constants.kv_dim * 2;
                k_out.* = self.arena.alloc_raw(bytes);
                v_out.* = self.arena.alloc_raw(bytes);
                ks_out.* = null;
                vs_out.* = null;
            },
            .fp8 => {
                const data_bytes = cap * constants.kv_dim;
                const scale_bytes = cap * constants.kv_head_count * 2;
                k_out.* = self.arena.alloc_raw(data_bytes);
                v_out.* = self.arena.alloc_raw(data_bytes);
                ks_out.* = self.arena.alloc_raw(scale_bytes);
                vs_out.* = self.arena.alloc_raw(scale_bytes);
            },
            .q4 => {
                const data_bytes = cap * (constants.kv_dim / 2);
                const scale_bytes = cap * constants.kv_q4_scales_per_token * 2;
                k_out.* = self.arena.alloc_raw(data_bytes);
                v_out.* = self.arena.alloc_raw(data_bytes);
                ks_out.* = self.arena.alloc_raw(scale_bytes);
                vs_out.* = self.arena.alloc_raw(scale_bytes);
            },
        }
    }

    fn record_target_forward(self: *Engine, batch_size: i32, commit_dspark: bool, compute_lm_head: bool) void {
        tiger.assert(batch_size >= 1 and batch_size <= constants.mtp_block_size_max);
        const w = self.weights.?;
        const b_usize: usize = @intCast(batch_size);
        const moe_slotted: u8 = if (batch_size > 1) 1 else 0;

        cuda.squid_launch_embed_norm(w.embed.?, w.embed_norm.?, self.d_tokens, self.hidden, batch_size, self.stream);

        var block_count: i32 = 0;
        var has_prior: bool = false;
        var prev_mlp_pending: bool = false;

        var vi: u32 = 0;
        while (vi < constants.visit_count) : (vi += 1) {
            const v = self.schedule[vi];
            const l: u32 = v.physical_layer;
            const lw: *const weights_mod.LayerWeights = &w.layers[l];

            var pre_tap_out: ?[*]f32 = null;
            if (v.is_loop and l == constants.loop_start_layer) {
                if (v.pass_index == 0) {
                    pre_tap_out = self.prelude;
                } else {
                    cuda.squid_launch_extract_block(self.blocks, 4, 2, self.saved, batch_size, self.stream);
                    cuda.squid_launch_rmsnorm(self.hidden, null, self.prior_last, batch_size, constants.hidden_dim, constants.hidden_dim, constants.rms_eps, self.stream);
                    tiger.assert_cuda(cuda.squid_cuda_memcpy_d2d_async(@ptrCast(self.hidden), @ptrCast(self.prelude), b_usize * constants.hidden_dim * 4, self.stream));
                    block_count = 1;
                    has_prior = true;
                }
            }
            const use_prior: bool = v.is_loop and has_prior;

            if (commit_dspark) {
                if (vi == constants.mtp_target_visits[0]) {
                    pre_tap_out = self.taps;
                } else if (vi == constants.mtp_target_visits[1]) {
                    pre_tap_out = self.taps + b_usize * constants.hidden_dim;
                } else if (vi == constants.mtp_target_visits[2]) {
                    pre_tap_out = self.taps + 2 * b_usize * constants.hidden_dim;
                }
            }

            const distinct_inputs: bool = v.is_global or use_prior;
            const partial_dst: ?[*]f32 = if (l % constants.block_layer_count == 0) self.partial else null;
            const use_half_dense: bool = (lw.w_qrgs.kind != @intFromEnum(constants.WeightKind.bf16));

            var n0: ?*const anyopaque = null;
            var n1: ?*const anyopaque = null;
            var n2: ?*const anyopaque = null;
            var o0: ?[*]f32 = null;
            var o1: ?[*]f32 = null;
            var o2: ?[*]f32 = null;
            var oh0: ?*anyopaque = null;
            var oh1: ?*anyopaque = null;
            var oh2: ?*anyopaque = null;
            var n_out: i32 = 1;

            if (!distinct_inputs) {
                n0 = lw.attn_norm;
                o0 = self.attn_in[0];
                oh0 = if (use_half_dense) self.attn_in_h[0] else null;
                n_out = 1;
            } else if (v.is_global and !use_prior) {
                n0 = lw.q_in_norm;
                n1 = lw.k_in_norm;
                n2 = lw.v_in_norm;
                o0 = self.attn_in[0];
                o1 = self.attn_in[1];
                o2 = self.attn_in[2];
                oh0 = if (use_half_dense) self.attn_in_h[0] else null;
                oh1 = if (use_half_dense) self.attn_in_h[1] else null;
                oh2 = if (use_half_dense) self.attn_in_h[2] else null;
                n_out = 3;
            } else if (v.is_global) {
                o0 = self.attn_in[0];
                o1 = self.attn_in[1];
                o2 = self.attn_in[2];
                n_out = 3;
            } else {
                o0 = self.mlp_in;
                n_out = 1;
            }

            if (prev_mlp_pending) {
                const prev_lw: *const weights_mod.LayerWeights = &w.layers[self.schedule[vi - 1].physical_layer];
                cuda.squid_launch_mega_mlp_post_next_depth_norm(
                    if (batch_size > 1) self.moe_slot_out else self.moe_down,
                    moe_slotted,
                    prev_lw.mlp_sconv_w.?,
                    self.mlp_ring[vi - 1],
                    self.residual,
                    self.hidden,
                    pre_tap_out,
                    null,
                    partial_dst,
                    self.blocks,
                    block_count,
                    w.attn_queries[l - 1].?,
                    n0,
                    n1,
                    n2,
                    o0,
                    o1,
                    o2,
                    oh0,
                    oh1,
                    oh2,
                    n_out,
                    batch_size,
                    self.d_pos,
                    self.stream,
                );
                prev_mlp_pending = false;
            } else {
                cuda.squid_launch_fused_depth_norm(
                    self.hidden,
                    partial_dst,
                    self.blocks,
                    block_count,
                    if (l > 0) w.attn_queries[l - 1] else null,
                    n0,
                    n1,
                    n2,
                    o0,
                    o1,
                    o2,
                    oh0,
                    oh1,
                    oh2,
                    n_out,
                    batch_size,
                    self.stream,
                );
            }

            if (use_prior) {
                const loc: usize = l - constants.loop_start_layer;
                const xs: [3][*]f32 = .{
                    if (v.is_global) self.attn_in[0] else self.mlp_in,
                    if (v.is_global) self.attn_in[1] else self.mlp_in,
                    if (v.is_global) self.attn_in[2] else self.mlp_in,
                };
                const outs: [3][*]f32 = .{ self.attn_in[0], self.attn_in[1], self.attn_in[2] };
                const outs_h: [3]?*anyopaque = .{ self.attn_in_h[0], self.attn_in_h[1], self.attn_in_h[2] };
                const cp_q: [3]?[*]const f32 = .{
                    w.cross_pass_queries.? + (2 * loc) * constants.hidden_dim,
                    w.cross_pass_kv_queries.? + (2 * loc) * constants.hidden_dim,
                    w.cross_pass_kv_queries.? + (2 * loc + 1) * constants.hidden_dim,
                };
                const cws: [3]?*const cuda.CControllerWeights = .{ &lw.ctrl[0], &lw.ctrl[1], &lw.ctrl[2] };
                const pnorms: [3]?*const anyopaque = .{
                    if (v.is_global) lw.q_in_norm else lw.attn_norm,
                    if (v.is_global) lw.k_in_norm else lw.attn_norm,
                    if (v.is_global) lw.v_in_norm else lw.attn_norm,
                };
                if (w.mode == .nvfp4) {
                    cuda.squid_launch_controller_mega_nvfp4(
                        &xs,
                        self.saved,
                        self.prior_last,
                        &cp_q,
                        &cws,
                        &pnorms,
                        &outs,
                        if (use_half_dense) &outs_h else null,
                        3,
                        batch_size,
                        self.stream,
                    );
                } else {
                    cuda.squid_launch_controller_merge_batched(
                        &xs,
                        self.saved,
                        self.prior_last,
                        &cp_q,
                        &cws,
                        &pnorms,
                        &outs,
                        if (use_half_dense) &outs_h else null,
                        3,
                        self.ctrl_msg,
                        self.ctrl_cat,
                        self.ctrl_d,
                        batch_size,
                        self.stream,
                    );
                }
            }

            const in_q = self.attn_in[0];
            const in_k = if (distinct_inputs) self.attn_in[1] else self.attn_in[0];
            const in_v = if (distinct_inputs) self.attn_in[2] else self.attn_in[0];
            const in_qh = self.attn_in_h[0];
            const in_kh = if (distinct_inputs) self.attn_in_h[1] else self.attn_in_h[0];
            const in_vh = if (distinct_inputs) self.attn_in_h[2] else self.attn_in_h[0];

            if (lw.w_qrgs.kind == @intFromEnum(constants.WeightKind.nvfp4)) {
                cuda.squid_launch_qkvrg_fused(&lw.w_qrgs, &lw.w_k, &lw.w_v, in_qh, in_kh, in_vh, self.qrgs, self.k_proj, self.v_proj, batch_size, self.stream);
            } else if (lw.w_qrgs.kind == @intFromEnum(constants.WeightKind.fp8)) {
                cuda.squid_launch_linear_h(&lw.w_qrgs, in_qh, self.qrgs, batch_size, 0, self.stream);
                cuda.squid_launch_linear_h(&lw.w_k, in_kh, self.k_proj, batch_size, 0, self.stream);
                cuda.squid_launch_linear_h(&lw.w_v, in_vh, self.v_proj, batch_size, 0, self.stream);
            } else {
                cuda.squid_launch_linear(&lw.w_qrgs, in_q, self.qrgs, batch_size, 0, self.stream);
                cuda.squid_launch_linear(&lw.w_k, in_k, self.k_proj, batch_size, 0, self.stream);
                cuda.squid_launch_linear(&lw.w_v, in_v, self.v_proj, batch_size, 0, self.stream);
            }

            if (v.is_global) {
                cuda.squid_launch_mega_global_sconv_prep_store(
                    self.qrgs,
                    self.k_proj,
                    self.v_proj,
                    in_k,
                    in_v,
                    lw.k_sconv_w.?,
                    lw.v_sconv_w.?,
                    lw.k_sconv_proj.?,
                    lw.v_sconv_proj.?,
                    self.k_ring[vi],
                    self.v_ring[vi],
                    lw.q_clip_scale.?,
                    lw.rel_clip_scale.?,
                    self.k_cache[vi].?,
                    self.v_cache[vi].?,
                    self.k_scale[vi],
                    self.v_scale[vi],
                    @intFromEnum(self.kv_format),
                    self.kv_cap[vi],
                    batch_size,
                    self.d_pos,
                    self.stream,
                );
            } else {
                cuda.squid_launch_mega_local_sconv_prep_store(
                    self.qrgs,
                    self.k_proj,
                    self.v_proj,
                    lw.k_sconv_w.?,
                    lw.v_sconv_w.?,
                    self.k_ring[vi],
                    self.v_ring[vi],
                    lw.q_clip_scale.?,
                    lw.rel_clip_scale.?,
                    self.k_cache[vi].?,
                    self.v_cache[vi].?,
                    self.kv_cap[vi],
                    batch_size,
                    self.d_pos,
                    self.stream,
                );
            }

            cuda.squid_launch_relative_attention_gated(
                self.qrgs,
                lw.rel_logits_proj.?,
                self.k_cache[vi].?,
                self.v_cache[vi].?,
                self.k_scale[vi],
                self.v_scale[vi],
                if (v.is_global) @intFromEnum(self.kv_format) else @intFromEnum(constants.KVFormat.bf16),
                self.split_buf,
                if (use_half_dense) null else self.attn_gated,
                if (use_half_dense) self.attn_gated_h else null,
                self.kv_cap[vi],
                if (v.is_global) 0 else constants.swa_window_tokens,
                batch_size,
                self.d_pos,
                self.stream,
            );

            if (use_half_dense) {
                cuda.squid_launch_linear_h(&lw.w_o, self.attn_gated_h, self.o_proj, batch_size, 0, self.stream);
            } else {
                cuda.squid_launch_linear(&lw.w_o, self.attn_gated, self.o_proj, batch_size, 0, self.stream);
            }

            const is_boundary: bool = (l % constants.block_layer_count == 0);
            cuda.squid_launch_mega_attn_post_mlp_in(
                self.o_proj,
                lw.attn_sconv_w.?,
                self.attn_ring[vi],
                self.hidden,
                self.partial,
                self.blocks,
                block_count,
                if (is_boundary) 1 else 0,
                self.residual,
                w.mlp_queries[l].?,
                if (use_prior) null else lw.mlp_norm,
                if (use_prior) self.mlp_in else self.mlp_normed,
                if (use_prior) null else self.mlp_normed_h,
                batch_size,
                self.d_pos,
                self.stream,
            );
            if (is_boundary) block_count += 1;

            if (use_prior) {
                const loc: usize = l - constants.loop_start_layer;
                const xs1: [1][*]f32 = .{self.mlp_in};
                const outs1: [1][*]f32 = .{self.mlp_normed};
                const outs_h1: [1]?*anyopaque = .{self.mlp_normed_h};
                const cp_q1: [3]?[*]const f32 = .{ w.cross_pass_queries.? + (2 * loc + 1) * constants.hidden_dim, null, null };
                const cws1: [3]?*const cuda.CControllerWeights = .{ &lw.ctrl[3], null, null };
                const pnorms1: [3]?*const anyopaque = .{ lw.mlp_norm, null, null };
                if (w.mode == .nvfp4) {
                    cuda.squid_launch_controller_mega_nvfp4(&xs1, self.saved, self.prior_last, &cp_q1, &cws1, &pnorms1, &outs1, &outs_h1, 1, batch_size, self.stream);
                } else {
                    cuda.squid_launch_controller_merge_batched(&xs1, self.saved, self.prior_last, &cp_q1, &cws1, &pnorms1, &outs1, &outs_h1, 1, self.ctrl_msg, self.ctrl_cat, self.ctrl_d, batch_size, self.stream);
                }
            }

            if (w.mode == .nvfp4) {
                cuda.squid_launch_route_topk_nvfp4(
                    self.mlp_normed_h,
                    &lw.router_nvfp4,
                    lw.router_bias.?,
                    self.router_logits,
                    self.route_ids,
                    self.route_w,
                    if (batch_size > 1) self.exp_groups else null,
                    batch_size,
                    self.stream,
                );
            } else {
                cuda.squid_launch_route_topk(
                    self.mlp_normed,
                    lw.router_w.?,
                    lw.router_bias.?,
                    self.router_logits,
                    self.route_ids,
                    self.route_w,
                    if (batch_size > 1) self.exp_groups else null,
                    batch_size,
                    self.stream,
                );
            }
            cuda.squid_launch_moe_gate_up_h(&lw.w13, self.mlp_normed_h, self.route_ids, self.exp_groups, self.moe_act_h, batch_size, self.stream);
            cuda.squid_launch_moe_down_h(&lw.w2, self.moe_act_h, self.route_ids, self.route_w, self.exp_groups, if (batch_size > 1) self.moe_slot_out else self.moe_down, batch_size, self.stream);

            if (vi == 17 or vi == 29) {
                cuda.squid_launch_mega_mlp_post_res(
                    if (batch_size > 1) self.moe_slot_out else self.moe_down,
                    moe_slotted,
                    lw.mlp_sconv_w.?,
                    self.mlp_ring[vi],
                    self.residual,
                    self.hidden,
                    batch_size,
                    self.d_pos,
                    self.stream,
                );
                prev_mlp_pending = false;
            } else {
                prev_mlp_pending = true;
            }
        }

        // Visit 47 final MLP post + output depth attention + final norm (+ tap 3)
        cuda.squid_launch_mega_mlp_post_next_depth_norm(
            if (batch_size > 1) self.moe_slot_out else self.moe_down,
            moe_slotted,
            w.layers[self.schedule[constants.visit_count - 1].physical_layer].mlp_sconv_w.?,
            self.mlp_ring[constants.visit_count - 1],
            self.residual,
            self.hidden,
            null,
            if (commit_dspark) (self.taps + 3 * b_usize * constants.hidden_dim) else null,
            null,
            self.blocks,
            block_count,
            w.output_query.?,
            w.final_norm,
            null,
            null,
            self.mlp_normed,
            null,
            null,
            null,
            null,
            null,
            1,
            batch_size,
            self.d_pos,
            self.stream,
        );

        // Overlap dspark_commit on aux_stream concurrently with lm_head_argmax on stream!
        if (compute_lm_head) {
            if (commit_dspark) {
                tiger.assert_cuda(cuda.squid_cuda_event_record(self.fork_event, self.stream));
                tiger.assert_cuda(cuda.squid_cuda_stream_wait_event(self.aux_stream, self.fork_event));
                cuda.squid_launch_dspark_commit(
                    &w.dspark,
                    self.taps,
                    &self.ds_k_cache,
                    &self.ds_v_cache,
                    &self.ds_k_scale,
                    &self.ds_v_scale,
                    &self.ds_kv_cap,
                    @intFromEnum(self.kv_format),
                    self.ds_f_ring.?,
                    self.ds_mix,
                    self.ds_kv,
                    batch_size,
                    self.d_pos,
                    self.aux_stream,
                );
                tiger.assert_cuda(cuda.squid_cuda_event_record(self.join_event, self.aux_stream));
            }
            cuda.squid_launch_lm_head_argmax(
                &w.unembed,
                self.mlp_normed,
                self.d_next_tokens,
                null,
                self.lm_blk_val,
                self.lm_blk_idx,
                batch_size,
                self.stream,
            );
            if (commit_dspark) {
                tiger.assert_cuda(cuda.squid_cuda_stream_wait_event(self.stream, self.join_event));
            }
        } else if (commit_dspark) {
            cuda.squid_launch_dspark_commit(
                &w.dspark,
                self.taps,
                &self.ds_k_cache,
                &self.ds_v_cache,
                &self.ds_k_scale,
                &self.ds_v_scale,
                &self.ds_kv_cap,
                @intFromEnum(self.kv_format),
                self.ds_f_ring.?,
                self.ds_mix,
                self.ds_kv,
                batch_size,
                self.d_pos,
                self.stream,
            );
        }
    }

    fn ensure_b1_graph(self: *Engine) void {
        if (self.b1_graph_exec != null or !self.use_graphs) return;
        tiger.assert_cuda(cuda.squid_cuda_graph_begin(self.stream));
        self.record_target_forward(1, false, true);
        cuda.squid_launch_inc_pos_and_copy_tok(self.d_pos, &self.d_tokens[0], &self.d_next_tokens[0], self.stream);
        tiger.assert_cuda(cuda.squid_cuda_graph_end_instantiate(self.stream, &self.b1_graph_exec));
        tiger.assert(self.b1_graph_exec != null);
    }

    fn ensure_prefill_b8_graph(self: *Engine, commit_dspark: bool) void {
        const slot: usize = if (commit_dspark) 1 else 0;
        if (self.prefill_b8_graph_exec[slot] != null or !self.use_graphs) return;
        tiger.assert_cuda(cuda.squid_cuda_graph_begin(self.stream));
        self.record_target_forward(constants.mtp_block_size_max, commit_dspark, false);
        cuda.squid_launch_inc_pos_by(self.d_pos, constants.mtp_block_size_max, self.stream);
        tiger.assert_cuda(cuda.squid_cuda_graph_end_instantiate(self.stream, &self.prefill_b8_graph_exec[slot]));
        tiger.assert(self.prefill_b8_graph_exec[slot] != null);
    }

    fn ensure_spec_graph(self: *Engine, num_drafts: u32) void {
        tiger.assert(num_drafts >= 1 and num_drafts <= constants.mtp_draft_count_max);
        if (self.spec_graph_exec[num_drafts] != null or !self.use_graphs) return;
        const w = self.weights.?;
        const batch: i32 = @intCast(num_drafts + 1);
        const ds_ks_const: [constants.mtp_layer_count]?*const anyopaque = .{ self.ds_k_scale[0], self.ds_k_scale[1], self.ds_k_scale[2] };
        const ds_vs_const: [constants.mtp_layer_count]?*const anyopaque = .{ self.ds_v_scale[0], self.ds_v_scale[1], self.ds_v_scale[2] };

        tiger.assert_cuda(cuda.squid_cuda_graph_begin(self.stream));
        cuda.squid_launch_dspark_draft(
            w.embed.?,
            w.embed_norm.?,
            &w.dspark,
            self.d_anchor,
            self.d_pos,
            @intCast(num_drafts),
            &self.ds_k_cache,
            &self.ds_v_cache,
            &ds_ks_const,
            &ds_vs_const,
            &self.ds_kv_cap,
            @intFromEnum(self.kv_format),
            self.split_buf,
            self.ds_f_ring.?,
            self.ds_ws,
            self.d_drafts,
            self.stream,
        );
        cuda.squid_launch_assemble_verify_tokens(self.d_anchor, self.d_drafts, self.d_tokens, @intCast(num_drafts), self.stream);
        self.record_target_forward(batch, true, true);
        cuda.squid_launch_spec_verify(self.d_next_tokens, self.d_drafts, batch, self.d_accepted, self.d_num_acc, self.d_pos, self.d_anchor, self.stream);
        tiger.assert_cuda(cuda.squid_cuda_graph_end_instantiate(self.stream, &self.spec_graph_exec[num_drafts]));
        tiger.assert(self.spec_graph_exec[num_drafts] != null);
    }
};

fn compute_runtime_vram_bytes(
    schedule: *const [constants.visit_count]constants.VisitInfo,
    context_tokens_max: u32,
    kv_format: constants.KVFormat,
    has_dspark: bool,
    expert_groups_bytes: u64,
) u64 {
    const a = constants.vram_alignment_bytes;
    var total: u64 = 0;

    var vi: u32 = 0;
    while (vi < constants.visit_count) : (vi += 1) {
        const is_glob = schedule[vi].is_global;
        const cap: u32 = if (is_glob) context_tokens_max else (constants.swa_window_tokens + constants.mtp_block_size_max);
        const vfmt: constants.KVFormat = if (is_glob) kv_format else .bf16;
        total += constants.kv_cache_pair_vram_bytes(cap, vfmt);
        total += 2 * tiger.align_up_u64(constants.sconv_ring_slots * constants.kv_dim * 4, a);
        total += 2 * tiger.align_up_u64(constants.sconv_ring_slots * constants.hidden_dim * 4, a);
    }

    if (has_dspark) {
        var ml: u32 = 0;
        while (ml < constants.mtp_layer_count) : (ml += 1) {
            const is_loc = constants.is_mtp_local_layer(ml);
            const cap: u32 = if (is_loc)
                (constants.swa_window_tokens + constants.mtp_block_size_max)
            else if (context_tokens_max > 524_288)
                65_536
            else
                context_tokens_max;
            const lfmt: constants.KVFormat = if (is_loc) .bf16 else kv_format;
            total += constants.kv_cache_pair_vram_bytes(cap, lfmt);
        }
        total += tiger.align_up_u64(constants.sconv_ring_slots * constants.hidden_dim * 4, a);
    }

    total += tiger.align_up_u64(constants.split_buffer_floats * 4, a);

    const b: u64 = constants.mtp_block_size_max;
    total += tiger.align_up_u64(4, a); // d_pos
    total += tiger.align_up_u64(b * 4, a); // d_tokens
    total += tiger.align_up_u64(b * 4, a); // d_next_tokens
    total += tiger.align_up_u64(4, a); // d_anchor
    total += tiger.align_up_u64(constants.mtp_draft_count_max * 4, a); // d_drafts
    total += tiger.align_up_u64(b * 4, a); // d_accepted
    total += tiger.align_up_u64(4, a); // d_num_acc

    total += 3 * tiger.align_up_u64(b * constants.hidden_dim * 4, a); // hidden, prelude, partial
    total += tiger.align_up_u64(b * 4 * constants.hidden_dim * 4, a); // blocks
    total += 2 * tiger.align_up_u64(b * constants.hidden_dim * 4, a); // saved, prior_last
    total += 3 * (tiger.align_up_u64(b * constants.hidden_dim * 4, a) + tiger.align_up_u64(b * constants.hidden_dim * 2, a));

    total += tiger.align_up_u64(b * constants.qrgs_out_dim * 4, a);
    total += 2 * tiger.align_up_u64(b * constants.kv_dim * 4, a);
    total += tiger.align_up_u64(b * constants.query_dim * 4, a);
    total += tiger.align_up_u64(b * constants.query_dim * 2, a);
    total += 4 * tiger.align_up_u64(b * constants.hidden_dim * 4, a); // o_proj, residual, mlp_in, mlp_normed
    total += tiger.align_up_u64(b * constants.hidden_dim * 2, a); // mlp_normed_h
    total += tiger.align_up_u64(3 * b * constants.hidden_dim * 4, a); // ctrl_msg
    total += tiger.align_up_u64(3 * b * 2 * constants.hidden_dim * 4, a); // ctrl_cat
    total += tiger.align_up_u64(3 * b * constants.gate_rank * 4, a); // ctrl_d
    total += tiger.align_up_u64(b * constants.expert_count * 4, a); // router_logits
    total += 2 * tiger.align_up_u64(b * constants.top_k_experts * 4, a); // route_ids, route_w
    total += tiger.align_up_u64(expert_groups_bytes, a);
    total += tiger.align_up_u64(b * constants.top_k_experts * constants.intermediate_dim * 2, a); // moe_act_h
    total += tiger.align_up_u64(b * constants.hidden_dim * 4, a); // moe_down
    total += tiger.align_up_u64(b * constants.top_k_experts * constants.hidden_dim * 4, a); // moe_slot_out
    total += tiger.align_up_u64(4 * b * constants.hidden_dim * 4, a); // taps
    total += tiger.align_up_u64(3 * b * constants.hidden_dim * 4, a); // ds_mix
    total += tiger.align_up_u64(b * 2 * constants.kv_dim * 4, a); // ds_kv
    total += tiger.align_up_u64(constants.dspark_workspace_floats * 4, a); // ds_ws
    total += 2 * tiger.align_up_u64(b * 1024 * 4, a); // lm_blk_val, lm_blk_idx

    return total;
}
