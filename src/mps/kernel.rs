use std::collections::HashMap;

pub struct KernelCache {
    pipelines: HashMap<String, metal::ComputePipelineState>,
}

impl KernelCache {
    pub fn new(device: &metal::Device, library: &metal::Library) -> Self {
        let mut pipelines = HashMap::new();
        let kernel_names = [
            "add_f32", "sub_f32", "mul_f32", "div_f32",
            "matmul_f32", "matmul_i32", "matmul_u32",
            "matmul_i16", "matmul_u16", "matmul_i8", "matmul_u8",
        ];
        for name in &kernel_names {
            if let Ok(func) = library.get_function(name, None) {
                if let Ok(pso) = device.new_compute_pipeline_state_with_function(&func) {
                    pipelines.insert(name.to_string(), pso);
                }
            }
        }
        KernelCache { pipelines }
    }

    pub fn get(&self, name: &str) -> Option<&metal::ComputePipelineState> {
        self.pipelines.get(name)
    }
}

pub fn compile_library(device: &metal::Device) -> Option<metal::Library> {
    let source = crate::mps::shaders::SHADER_SOURCE;
    let options = metal::CompileOptions::new();
    device.new_library_with_source(source, &options).ok()
}

pub fn dispatch_1d(
    queue: &metal::CommandQueueRef,
    pso: &metal::ComputePipelineStateRef,
    buffers: &[&metal::BufferRef],
    count: u64,
) {
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pso);
    for (i, buf) in buffers.iter().enumerate() {
        encoder.set_buffer(i as u64, Some(*buf), 0);
    }

    let thread_group_size = pso.max_total_threads_per_threadgroup();
    let w = thread_group_size.min(1024);
    let groups = count.div_ceil(w);
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(groups, 1, 1),
        metal::MTLSize::new(w, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn dispatch_2d(
    queue: &metal::CommandQueueRef,
    pso: &metal::ComputePipelineStateRef,
    buffers: &[&metal::BufferRef],
    constants: &[u32],
    width: u64,
    height: u64,
) {
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pso);
    let mut buf_idx = 0;
    for buf in buffers {
        encoder.set_buffer(buf_idx as u64, Some(*buf), 0);
        buf_idx += 1;
    }
    for &val in constants {
        let data = val.to_ne_bytes();
        let buf = queue.device().new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            data.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buf_idx as u64, Some(&buf), 0);
        buf_idx += 1;
    }

    let w = pso.max_total_threads_per_threadgroup().min(16);
    let h = pso.max_total_threads_per_threadgroup().min(16);
    let gw = width.div_ceil(w);
    let gh = height.div_ceil(h);
    encoder.dispatch_thread_groups(
        metal::MTLSize::new(gw, gh, 1),
        metal::MTLSize::new(w, h, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}