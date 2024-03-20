### This code collects emojis from different categories, 
### converts them to images, extracts features using SIFT 
### and gray histogram, trains a classifier using SVM, 
### tests the classifier with emojis and human face emotion
### datasets, and predicts the categories of unseen emojis 
### and human face emotions.

import numpy as np
import cv2
import os
import skimage.io
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image, ImageFont, ImageDraw
from IPython.display import display
import urllib.request
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern, hog
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV


# Task 1: Collect emojis, 20-30 categories

emoji_categories = {
    'smiling': ['😀', '😃', '😄', '😁', '😆'],
    'affectionate': ['🥰', '😍', '🤩', '😘', '😚'],
    'tongue': ['😋', '😛', '😜', '🤪', '😝'],
    'hands': ['🤗', '🤭', '🫢', '🫣', '🤫'],
    'neutral': ['🙃', '🫠', '🤐', '🤨', '😐'],
    'skeptical': ['😒', '🙄', '😬', '😮‍💨', '🤥'],
    'sleepy': ['😌', '😔', '😪', '😴'],
    'unwell': ['😷', '🤒', '🤕'],
    'disgusting': ['🤮','🤢','🥴'],
    'crying': ['😢','😭','🥲'],
    'scared': ['😱','😨','😰'],
    'joy': ['😈', '👿', '🤡', '👻', '👽'],
    'glasses': ['🥸','😎','🤓','🧐'],
    'doctor': ['🧑‍⚕️', '👨‍⚕️', '👩‍⚕️'],
    'student': ['🧑‍🎓', '👨‍🎓', '👩‍🎓'],
    'farmer': ['🧑‍🌾', '👨‍🌾', '👩‍🌾'],
    'cook':['🧑‍🍳', '👨‍🍳', '👩‍🍳'],
    'police':['👮‍♀️','👮‍♂️','👮'],
    'teacher':['👩‍🏫','🧑‍🏫','👨‍🏫'],
    'fruit': ['🍇', '🍈', '🍉', '🍊', '🍋'],
    'vegetable': ['🍆', '🥔', '🥕', '🌽', '🌶️'],
    'dessert': ['🍦', '🍧', '🍨', '🍩', '🍪', '🎂'],
    'drink': ['🍾', '🍷', '🍺', '🍻', '🥂', '🥃'],
    'award': ['🎖️', '🏆', '🏅', '🥇', '🥈', '🥉'],
    'geography': ['🌍', '🌎', '🌏', '🏔️', '⛰️', '🌋'],
    }

# convert the emojis to images

def emoji_to_image(emoji: str, size: int = 64) -> np.ndarray:
    image = Image.new("L", (64,64), (255))
    # please change file location of NotoEmoji.ttf
    font = ImageFont.truetype("C:/Users/zoezh/Desktop/COMP4423/A1/submission/NotoEmoji.ttf", 60, encoding='unic')
    draw = ImageDraw.Draw(image)
    draw.textbbox(xy=[0,0], text=emoji, font=font)
    draw.text((0, 0), emoji, fill=(0), font=font)
    return np.array(image)

# Save img_np_array, label number index, label name into list
X = [] # emoji_img_list
y = [] # label number index
keys = [] # label list
emojis = []
# values = {}
for key, value in emoji_categories.items():
    keys.append(key)
    # values[key] = value
    for i in value:
        # print(i)
        emojis.append(i)
        img = emoji_to_image(i)
        img_array = np.array(img).flatten()
        X.append(img_array)
        y.append(keys.index(key)) # append label(number)
        # len += 1

# print(emojis)

labels = []
for j in y:
    labels.append(keys[j])
# print(labels)

X = np.array(X)
y = np.array(y)

# split image, index, emoji, label into train dataset and val dataset
X_train, X_val, y_train, y_val, emoji_train, emoji_val, label_train, label_val= train_test_split(X, y, emojis, labels, test_size=0.2, random_state=42)

# print(labels_train)
# print(labels_val)

# Task 2: extract features
feat_train = []
feat_val = []
sift = cv2.SIFT_create()

def get_feat(img):
    return hog(img)
    # return sift_feat(img)
    # return gray_histogram(img)

def calc_distance(x, y):
    return L2_distance(x, y)
#     return L2_distance_sift(x, y)

def sift_feat(img):
    kps, des =  sift.detectAndCompute(img, None)
    responses = [kp.response for kp in kps]
    order = np.argsort(responses)[::-1]
    return np.array(des[order[:30]])

def gray_histogram(img: np.array, norm: bool = True) -> np.array:
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = np.array([len(img[img == i]) for i in range(256)])
    if norm:
        return hist / np.size(img)
    return hist

def L2_distance(x, y):
    return ((x - y) ** 2).sum() ** 0.5

def L2_distance_sift(x, y):
    dist = ((x[:, None] - y[None, :])**2).sum(axis=-1).min(axis=-1)
    dist.sort()
    return dist[:15].mean()

for sample in X_train:
    feat_train = get_feat(X_train)

for sample in X_val:
    feat_val = get_feat(X_val)


# Task 3: Train a classifier
# Train with SVM
print('Testing original data with SVM')
print("------------------------------")
model = SVC(kernel= 'linear')
model.fit(X_train, y_train)
# test with SVM (self-values)
y_pred = model.predict(X_val) 
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

scores = cross_val_score(model, X, y, cv=3)  # 5-fold cross-validation
print(f'Cross-Validation Accuracy: {np.mean(scores) * 100:.2f}%')

category_list = list(emoji_categories)
for val, pred, l in zip(emoji_val, y_pred, label_val):
    print(val, 'with label', l,'is predicted as', category_list[pred])



# Task 4: Test the classifier using unseen emojis created by Emoji Kitchen
print('\n\nTesting model using unseen emoji')
print('--------------------------------')
# test unseen emoji from folder located in 'C:/Users/zoezh/Desktop/COMP4423/A1/submission/unseen'
unseen_path = 'C:/Users/zoezh/Desktop/COMP4423/A1/submission/unseen'

unseen_img_list = []
unseen_img_fname = []
# Iterate over the files in the folder
for file in os.listdir(unseen_path):
    # Check if the file is an image (you can modify the condition as per your file extensions)
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Construct the full file path
        file_path = os.path.join(unseen_path, file)

        # Read the image
        image = cv2.imread(file_path)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Check if the image was successfully read
        if image is not None:
            # Convert the image to a NumPy array
            # image_array = np.array(image)
            # Convert the image to 64 * 64 image array and then convert color into gray, then flatten
            image = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_BGR2GRAY).flatten()

            # Display the shape of the image array
            print("Image:", file)
            print("Image shape:", image.shape)
            # display image
            # plt.imshow(image_rgb)
            # plt.axis('off')  # Turn off the axis labels
            # plt.show()
        else:
            print("Failed to read the image:", file)

    
    unseen_img_list.append(image)

unseen_img_list = np.array(unseen_img_list)

# test with model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
unseen_pred = model.predict(unseen_img_list) 
print('Result:')
unseen_pred_result = []
for pred in unseen_pred:
    unseen_pred_result.append(category_list[pred])
print(unseen_pred_result)


# Task 5: Test the model using human face emotions datasets
print('\n\nTesting model using human face')
print('------------------------------')
face_path = 'C:/Users/zoezh/Desktop/COMP4423/A1/submission/face'

face_img_list = []
face_img_fname = []
# Iterate over the files in the folder
for file in os.listdir(face_path):
    # Check if the file is an image
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Construct the full file path
        file_path = os.path.join(face_path, file)

        # Read the image using OpenCV
        face_image = cv2.imread(file_path)

        # Check if the image was successfully read
        if image is not None:
            # Convert the image to a NumPy array
            image_array = np.array(image).flatten()

            # Display the shape of the image array
            print("Image:", file)
            # unseen_img_fname.append(file)
            print("Image shape:", image.shape)
            # display(image)
            # plt.imshow(image_rgb)
            # plt.axis('off')  # Turn off the axis labels
            # plt.show()
        else:
            print("Failed to read the image:", file)
    
    face_img_list.append(image)

face_img_list = np.array(face_img_list)


svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
face_pred = model.predict(face_img_list) 

face_pred_result = []
for pred in face_pred:
    face_pred_result.append(category_list[pred])
print(face_pred_result)