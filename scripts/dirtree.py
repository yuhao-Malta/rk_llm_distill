import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class DirectoryExporter:
    def __init__(self, root):
        self.root = root
        self.root.title("Windows 11 目录结构导出工具")
        self.root.geometry("600x450")
        self.root.resizable(True, True)

        # 设置主题风格
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.setup_ui()

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 目录选择部分
        dir_frame = ttk.LabelFrame(main_frame, text="目录选择", padding="5")
        dir_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        self.dir_path = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.dir_path, width=50).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(dir_frame, text="浏览...", command=self.browse_directory).grid(row=0, column=1)

        # 选项部分
        options_frame = ttk.LabelFrame(main_frame, text="导出选项", padding="5")
        options_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(options_frame, text="最大层级深度:").grid(row=0, column=0, sticky=tk.W)
        self.depth_var = tk.StringVar(value="全部")
        depth_combo = ttk.Combobox(options_frame, textvariable=self.depth_var,
                                   values=["全部", "1", "2", "3", "4", "5"])
        depth_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        ttk.Label(options_frame, text="输出文件:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.output_path = tk.StringVar(value="directory_structure.txt")
        ttk.Entry(options_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, pady=(10, 0))

        # 按钮部分
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=10)

        ttk.Button(button_frame, text="导出目录结构", command=self.export).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="清除", command=self.clear).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="退出", command=self.root.quit).grid(row=0, column=2, padx=5)

        # 状态部分
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky=tk.W)

        # 配置权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        dir_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)

    def browse_directory(self):
        directory = filedialog.askdirectory(title="选择要导出结构的目录")
        if directory:
            self.dir_path.set(directory)

    def export(self):
        directory = self.dir_path.get()
        if not directory or not os.path.isdir(directory):
            messagebox.showerror("错误", "请选择有效的目录")
            return

        output_file = self.output_path.get()
        if not output_file:
            messagebox.showerror("错误", "请输入输出文件名")
            return

        try:
            max_depth = None if self.depth_var.get() == "全部" else int(self.depth_var.get())
            self.status_var.set("正在导出...")
            self.root.update()

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"目录结构: {directory}\n")
                f.write("=" * 50 + "\n\n")
                self.scan_directory(f, directory, 0, max_depth)

            self.status_var.set(f"导出完成! 结果保存在: {output_file}")
            messagebox.showinfo("成功", f"目录结构已成功导出到 {output_file}")

        except Exception as e:
            messagebox.showerror("错误", f"导出过程中发生错误: {str(e)}")
            self.status_var.set("导出失败")

    def scan_directory(self, file, path, current_depth, max_depth):
        if max_depth is not None and current_depth > max_depth:
            return

        indent = "    " * current_depth
        dir_name = os.path.basename(path) if current_depth > 0 else path

        if current_depth == 0:
            file.write(f"{dir_name}/\n")
        else:
            file.write(f"{indent}├── {dir_name}/\n")

        try:
            items = os.listdir(path)
            dirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
            files = [f for f in items if os.path.isfile(os.path.join(path, f))]

            for d in dirs:
                self.scan_directory(file, os.path.join(path, d), current_depth + 1, max_depth)

            file_indent = "    " * (current_depth + 1)
            for f in files:
                file.write(f"{file_indent}├── {f}\n")

        except PermissionError:
            file.write(f"{indent}    └── [权限被拒绝]\n")

    def clear(self):
        self.dir_path.set("")
        self.output_path.set("directory_structure.txt")
        self.depth_var.set("全部")
        self.status_var.set("就绪")


if __name__ == "__main__":
    root = tk.Tk()
    app = DirectoryExporter(root)
    root.mainloop()
