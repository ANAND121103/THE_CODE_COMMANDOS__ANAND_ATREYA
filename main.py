import cv2
import easyocr

# Load the image
image = cv2.imread('/Users/anand/Library/CloudStorage/GoogleDrive-anandatreya21@gmail.com/My Drive/dataset/train/images')

# Create an EasyOCR reader
reader = easyocr.Reader(['en'])  # specify the language as English

# Use EasyOCR to extract text from the image
result = reader.readtext(image)

# Extract the text from the result
text = ''
for (bbox, text, prob) in result:
    text += text + ' '

# Print the extracted text
print(text)