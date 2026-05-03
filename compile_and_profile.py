import qai_hub
import onnx
import os
import sys

# =============================================================================
# USER CONFIGURATION
# =============================================================================

ONNX_DIR = "ONNX_DIR"
VIDEO_ONNX_NAME = "model.onnx"
DEVICE_NAME = "Dragonwing IQ-9075 EVK"

# Model input (NCTHW — DO NOT CHANGE)
BATCH = 1
C = 3
T = 16
H = 112
W = 112

# =============================================================================


def run_profile(model, device):
    """Submit a profiling job to QAI Hub and return the job ID."""
    profile_job = qai_hub.submit_profile_job(
        model=model,
        device=device,
        options="--max_profiler_iterations 100"
    )
    return profile_job.job_id


def compile_model(model, device, input_specs):
    options = (
        "--target_runtime qnn_context_binary "
        "--qairt_version 2.43 "
        "--output_names class_probs "
        "--force_channel_last_input video "
        "--qnn_options context_enable_graphs=resnet_2plus1d_float"
    )

    print(f"\nCompile options: {options}")

    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options=options
    )
    return compile_job.job_id


def main():

    if not ONNX_DIR:
        print("Error: 'ONNX_DIR' is not set.")
        sys.exit(1)

    VIDEO_ONNX_PATH = os.path.join(ONNX_DIR, VIDEO_ONNX_NAME)

    if not os.path.exists(VIDEO_ONNX_PATH):
        print(f"Error: '{VIDEO_ONNX_PATH}' not found.")
        sys.exit(1)

    print(f"Loading ONNX model from {VIDEO_ONNX_PATH}...")

    # 🔍 Validate ONNX (good practice)
    onnx_model = onnx.load(VIDEO_ONNX_PATH)
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid ✅")
    except onnx.checker.ValidationError as e:
        print("ONNX validation failed ❌")
        print(e)
        sys.exit(1)

    # 🚀 Upload model to AI Hub (IMPORTANT)
    print("Uploading ONNX model to AI Hub...")
    hub_model = qai_hub.upload_model(VIDEO_ONNX_PATH)

    device = qai_hub.Device(DEVICE_NAME)

    # ✅ CORRECT input spec (must match ONNX exactly)
    input_specs = {
        "video": ((BATCH, C, T, H, W), "float32")
    }

    print(f"\nUsing input spec: {input_specs}")

    # ==============================
    # Compile
    # ==============================
    print("\nSubmitting compilation job to QAI Hub...")

    compile_id = compile_model(
        model=hub_model,
        device=device,
        input_specs=input_specs,
    )

    print(f"Compilation job ID: {compile_id}")

    # ==============================
    # Wait + Profile
    # ==============================
    print("\nWaiting for compilation to finish...")

    compile_job = qai_hub.get_job(compile_id)
    target_model = compile_job.get_target_model()

    print("Compilation finished ✅")

    print("\nSubmitting profiling job...")

    profile_id = run_profile(target_model, device)

    print(f"Profiling job ID: {profile_id}")
    print("Done.")


if __name__ == "__main__":
    main()