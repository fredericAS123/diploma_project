from Qwen_inference import QwenInferenceWrapper
from webui_gradio import VideoChatWebUI

# 模型路径配置
MODEL_PATH = "/root/autodl-tmp/Qwen/Qwen2___5-VL-3B-Instruct"

def main():
    # 1. 初始化推理引擎 (Backend)
    # 这一步会加载模型，比较耗时
    print(">>> Initializing Inference Engine...")
    inference_engine = QwenInferenceWrapper(model_path=MODEL_PATH)
    
    # 2. 初始化 WebUI (Frontend)
    # 将推理引擎注入到 UI 类中
    print(">>> Initializing WebUI...")
    web_ui = VideoChatWebUI(inference_engine)
    
    # 3. 创建并启动 Gradio 界面
    demo = web_ui.create_interface()
    print(">>> Launching Gradio Server...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=6006, 
        share=False, 
        debug=True
    )

if __name__ == "__main__":
    main()