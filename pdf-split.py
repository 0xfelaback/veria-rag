import os
from pypdf import PdfReader, PdfWriter


def split_pdf_by_page_count(input_pdf_path, pages_per_split=10):
    """
    Splits an input PDF into multiple chunks of a specified page count.
    """
    # Ensure the input file exists
    if not os.path.exists(input_pdf_path):
        print(f"Error: The file '{input_pdf_path}' does not exist.")
        return

    # Read the original PDF
    reader = PdfReader(input_pdf_path)
    total_pages = len(reader.pages)

    # Extract base filename and extension
    base_name, _ = os.path.splitext(input_pdf_path)

    print(f"Processing '{input_pdf_path}' ({total_pages} total pages)...")

    # Iterate through the document in chunks of 10 pages
    for start_page in range(0, total_pages, pages_per_split):
        writer = PdfWriter()

        # Determine the end boundary for the current chunk
        end_page = min(start_page + pages_per_split, total_pages)

        # Add the range of pages to the writer
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        # Define the new file name (e.g., document_pages_1_to_10.pdf)
        output_filename = f"{base_name}_pages_{start_page + 1}_to_{end_page}.pdf"

        # Save the new PDF chunk
        with open(output_filename, "wb") as output_file:
            writer.write(output_file)

        print(f"Created: {output_filename}")


if __name__ == "__main__":
    # Replace with your actual file path
    target_pdf = "SBL_UPDATED_EMPLOYEE_HANDBOOK.pdf"

    # Run the split function
    split_pdf_by_page_count(target_pdf, pages_per_split=10)
