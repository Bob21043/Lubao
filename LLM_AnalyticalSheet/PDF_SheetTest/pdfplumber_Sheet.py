import pdfplumber
import os


# initialize LLM model
def extract_tables_with_pdfplumber(pdf_path, pages=None):
    table_list = []
    with pdfplumber.open(pdf_path) as pdf:
        page_list = range(len(pdf.pages)) if pages is None else pages

        for i in page_list:
            page = pdf.pages[i]
            try:
                tables = page.extract_tables()
                for ti, table in enumerate(tables):
                    if table:  # 過濾空表
                        table_list.append({
                            "page": i + 1,   # 轉成 1-based 頁碼
                            "index": ti + 1,
                            "array": table
                        })
            except Exception as e:
                table_list.append({
                    "page": i + 1,
                    "index": None,
                    "array": [],
                    "error": str(e)
                })
    return table_list

# 新增：存檔函式
def save_tables_to_txt(tables, output_path="extracted_tables.txt"):
    # 如果不存在就新建檔案，存在就覆蓋
    with open(output_path, "w", encoding="utf-8") as f:
        for table in tables:
            f.write(f"\n【第{table['page']}頁 表格{table['index']}】\n")
            for row in table["array"]:
                # 把 None 轉成空字串，避免顯示 None
                row_text = "\t".join([cell if cell else "" for cell in row])
                f.write(row_text + "\n")
    print(f"表格已存到 {os.path.abspath(output_path)}")

def main():

    pdf_path =r"C:\Users\medgreen\Desktop\LuBao\表格分析\自動化教育訓練手冊_部屬者_250917.pdf"

    # 擷取PDF的表格
    tables =extract_tables_with_pdfplumber(pdf_path)
    print(f"總共擷取到 {len(tables)} 張表格。")
    for table in tables:
        print(f"\n【第{table['page']}頁 表格{table['index']}】")
        for row in table["array"]:
            print(row)
    # 存檔
    # save_tables_to_txt(tables, output_path="extracted_tables.txt")

if __name__ == "__main__":
    main()
    