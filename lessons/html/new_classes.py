import re
from pathlib import Path
 
def transform_html(file_path):

    """Opens a file, transforms HTML, and saves the modified content.
 
    Args:

        file_path (str): The path to your HTML file.

    """
 
    with open(file_path, 'r', encoding='utf-8') as file:

        text = file.readlines()
        # text = text[1:-1]
        text = "".join(text)
 
    # Transformations using regular expressions 

    # text = re.sub(r'<p class=".*?"', '<p class="lesson-paragraph"', text)

    # text = re.sub(r'<h1 class=".*?"', '<h1 class="lesson-heading"', text)

    # text = re.sub(r'<h2 class=".*?"', '<h2 class="lesson-sub-heading"', text)

    text = re.sub(r'<code>', '<code class="lesson-code">', text)
    text = re.sub(r'<h1', '<h1 class="lesson-heading"', text)
    text = re.sub(r'<h2', '<h2 class="lesson-sub-heading"', text)
    text = re.sub(r'<li ', '<li class="lesson-li" ', text)
    text = re.sub(r'<p', '<p class="lesson-paragraph"', text)

    # text = re.sub(r'<a', '<a class="lesson-link"', text)

    # text = re.sub(r'<li class=".*?"', '<li class="lesson-li"', text)

    
    file_path = Path(file_path)
    out_file = file_path.with_name(file_path.name+'_css_classes.html')
    
    with open(out_file, 'w', encoding='utf-8') as file:

        file.write(text)
 
 
# Example usage

file_path = "lessons/html/tensor_introduction.html"  # Replace with the path to your file

transform_html(file_path)
