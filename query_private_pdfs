import os
import PyPDF2
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        num_pages = pdf_reader.numPages
        for page_num in range(num_pages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

def generate_query(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return query

def main():
    model_name = "gpt2"  # Use "gpt2" or specify a custom model if you have one
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    pdf_folder = "C:/Users/caspa/Desktop/Trialfolder/Documents"  # Update with your folder containing PDFs
    prompt = "Please enter your query: "
    
        user_input = input(prompt)
        if user_input.lower() == "exit":
            break
        query = generate_query(user_input, model, tokenizer)
        print("Generated Query:", query)

        for pdf_file in os.listdir(pdf_folder):
            if pdf_file.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, pdf_file)
                pdf_text = load_pdf_text(pdf_path)
                if query in pdf_text:
                    print(f"Match found in '{pdf_file}'")

if __name__ == "__main__":
    main()
