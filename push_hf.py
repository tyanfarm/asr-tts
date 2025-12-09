import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import zipfile
from huggingface_hub import HfApi

# --- CẤU HÌNH (SỬA Ở ĐÂY) ---
HF_TOKEN = ""  # Token (quyền WRITE)
REPO_ID = "tyanfarm/ai-mate-zip"       # Ví dụ: "namnguyen/my-tts-app"
REPO_TYPE = "model"                         # 'model', 'dataset', hoặc 'space'

# Đường dẫn đến file hoặc folder bạn muốn nén và upload
# Ví dụ 1: Chỉ file exe -> "dist/ung_dung.exe"
# Ví dụ 2: Cả folder build -> "dist/ung_dung_folder"
INPUT_PATH = "dist/ai_mate.exe" 

# Tên file zip bạn muốn tạo ra và tên trên Hugging Face
ZIP_NAME = "ai-mate-cpu.zip" 
# ----------------------------

def create_zip(input_path, output_zip):
    print(f"📦 Đang nén '{input_path}' thành '{output_zip}'...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.isfile(input_path):
            # Nếu input là 1 file đơn lẻ
            zipf.write(input_path, os.path.basename(input_path))
        elif os.path.isdir(input_path):
            # Nếu input là 1 folder (duyệt đệ quy)
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Tính đường dẫn tương đối để giữ cấu trúc folder trong zip
                    arcname = os.path.relpath(file_path, os.path.dirname(input_path))
                    zipf.write(file_path, arcname)
    
    # Kiểm tra dung lượng sau nén
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"✅ Đã nén xong. Dung lượng: {size_mb:.2f} MB")

def upload_to_hf(zip_file, repo_id, token):
    print(f"🚀 Đang upload '{zip_file}' lên '{repo_id}'...")
    api = HfApi(token=token)
    
    api.create_repo(repo_id=repo_id, repo_type=REPO_TYPE, exist_ok=True)
    
    try:
        api.upload_file(
            path_or_fileobj=zip_file,
            path_in_repo=zip_file, # Giữ nguyên tên file zip trên repo
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            commit_message=f"Upload {zip_file} (compressed build)"
        )
        print("✅ Upload thành công!")
        print(f"🔗 Link tải: https://huggingface.co/{repo_id}/resolve/main/{zip_file}")
    except Exception as e:
        print(f"❌ Lỗi khi upload: {e}")

if __name__ == "__main__":
    # 1. Kiểm tra file đầu vào
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Không tìm thấy đường dẫn: {INPUT_PATH}")
    else:
        try:
            # 2. Thực hiện nén
            if not os.path.exists(ZIP_NAME):
                create_zip(INPUT_PATH, ZIP_NAME)
            
            # 3. Thực hiện upload
            upload_to_hf(ZIP_NAME, REPO_ID, HF_TOKEN)
            
            # 4. Dọn dẹp (xóa file zip ở máy local sau khi up xong - tùy chọn)
            # os.remove(ZIP_NAME) 
            # print("🧹 Đã xóa file zip tạm trên máy.")
            
        except Exception as e:
            print(f"❌ Có lỗi xảy ra: {e}")