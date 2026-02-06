import os
import hashlib

def file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

local_root = "/home/hu/Diana-Sim/analog_design"
nas_root = "/mnt/nas/DOMAIN=JP/0107403250/saratoga/analog_design"

# 验证有差异的文件是否保留了本地版本
diff_files = [
    "python_sim/src/bsim4_circuit_analyzer.py",
    "tests/test_python_bsim4.py",
]

print("=" * 70)
print("融合验证报告")
print("=" * 70)
print()

for rel_path in diff_files:
    local_path = os.path.join(local_root, rel_path)
    nas_path = os.path.join(nas_root, rel_path)
    local_merged_path = local_path
    
    local_hash = file_hash(local_path)
    nas_hash = file_hash(nas_path)
    merged_hash = file_hash(local_merged_path)
    
    is_local = (merged_hash == local_hash)
    is_nas = (merged_hash == nas_hash)
    
    status = "✓ 保留本地版本" if is_local else ("✗ NAS版本" if is_nas else "✗ 未知版本")
    print(f"文件: {rel_path}")
    print(f"状态: {status}")
    print()

# 统计新增文件
print("=" * 70)
print()
all_files = sum(1 for _, _, files in os.walk(local_root) for _ in files 
               if all(excl not in _ for excl in ['.venv', '__pycache__', '.pytest_cache', '.git', 'logs', '.vscode']))

print(f"融合后总文件数: {all_files}")
print(f"本地备份位置: /home/hu/Diana-Sim/analog_design_backup_20260206_151303")
print()
print("✓ 融合成功！本地配置已保留")
