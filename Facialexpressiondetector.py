# Import the required modules
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# Function to detect facial expressions
def detect_emotion(image_path):
    try:
        # Read the image
        img = cv2.imread("D:\Sohel Datta\Happy boy.jpg")
        if img is None:
            print("Error: Image not found. Please check the file path.")
            return
        
        # Display the image using matplotlib
        plt.imshow(img[:, :, ::-1])  # Convert BGR to RGB for display
        plt.title("Input Image")
        plt.axis("off")
        plt.show()

        # Analyze the image for emotions using DeepFace
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        # Print the result
        print("\nEmotion Analysis Result:")
        print(result)

        # Display the dominant emotion
        dominant_emotion = result['dominant_emotion']
        print(f"\nDominant Emotion: {dominant_emotion}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Provide the path to your image
image_path = r"D:\Sohel Datta\IMP Docs\Sohel Datta Pic.jpg"

# Call the emotion detection function
detect_emotion(image_path)
