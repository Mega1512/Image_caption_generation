Image Caption Generation with ResNet50 and LSTM

Introduction

In this project, I developed an image caption generation system using ResNet50 for image encoding and LSTM for text generation. The system is designed to automatically generate descriptive captions for images, providing insights into both the deep learning process and web application development.

Understanding the Dataset

•	Dataset Used: Flickr8k
•	Components:
  o	Images: 8,000 images of varying content.
  o	Captions: Five different descriptions for each image.
•	Purpose: To train a model that can learn the relationship between image features and corresponding captions.

Data Preprocessing

•	Loading Descriptions: Extracted and cleaned captions from the provided text file.
•	Cleaning Process:
  o	Removed punctuation.
  o	Converted to lowercase.
  o	Filtered out non-alphabetical words.
•	Building Vocabulary: Created a vocabulary of words that appear frequently in the captions.
•	Image Preprocessing: Used ResNet50 to extract features from images by resizing and normalizing them.

Model Architecture
•	Image Encoding:
  o	Used ResNet50 (pre-trained on ImageNet) without the top layer.
  o	Extracted a 2048-dimensional feature vector for each image.
•	Caption Generation:
  o	Tokenized captions and converted them into sequences.
  o	Used an embedding layer, LSTM, and dense layers to generate captions.
  o	Combined image and text features using concatenation and addition layers.
  
Training the Model
•	Data Generator: Created a generator to yield batches of image features and corresponding caption sequences.
•	Training:
  o	Used early stopping and learning rate reduction on plateau for better training performance.
  o	Saved the best model based on training loss.

Building the Web Application
•	Flask Setup: Set up a Flask application to handle image uploads and caption generation.
•	Functionality:
  o	Uploaded images are processed and features extracted using ResNet50.
  o	Generated captions are displayed alongside the uploaded image.
•	User Interface: Used HTML, CSS, and Bootstrap for a user-friendly interface.

Containerization and Deploying the Application
•	Containerization: Used Docker to containerize the entire project and obtain docker image, offering benefits such as portability, isolation, scalability, and  versioning.
•	Deployment:  Utilized Render as a cloud platform for web application deployment.

Challenges and Solutions
•	Input Shape Issues: Ensured consistency in image dimensions across the model.
•	Memory Management: Implemented efficient data handling using generators.
•	Model Integration: Combined ResNet50 and LSTM models seamlessly for end-to-end caption generation.

Future Enhancements
•	Improved Model: Experiment with advanced architectures like Transformer-based models for better caption quality.
•	Dataset Expansion: Use larger and more diverse datasets to improve model generalization.

Skills acquired
•	Text processing
•	Image processing
•	CNN (ResNet50) and RNN (LSTM) model architecture design
•	Embedding layers
•	Model Training and optimisation 
•	Web Development
•	Model deployment
•	Problem solving and Debugging

Conclusion
This project provided valuable insights into the integration of computer vision and natural language processing. By building and deploying an image caption generation system, I enhanced my skills in deep learning, model optimization, and web application development, paving the way for more advanced projects in the future.

