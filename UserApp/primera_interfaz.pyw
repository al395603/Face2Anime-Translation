from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2

import numpy as np
import tensorflow as tf
from tensorflow import keras

from CycleGan import CycleGan, get_resnet_generator, get_discriminator, generator_loss_fn, discriminator_loss_fn


def saveImage(img1, img2, img3, img4):
    messagebox.showinfo("Images saved", "Images saved successfully.")
    img1.save("predicted_img_1.png")
    img2.save("predicted_img_2.png")
    img3.save("predicted_img_3.png")
    img4.save("predicted_img_4.png")

def generateImage(path):
    global btnGenerate

    input_img_size = (256, 256, 3)
    messagebox.showinfo("Generating images...", "Wait while images are being generated, it may take a few seconds.")

    # Image preprocessing.
    imgTF = tf.io.read_file(path)
    imgTF = tf.image.decode_jpeg(imgTF, channels=3)
    imgTF = tf.image.resize(imgTF, [input_img_size[0], input_img_size[1]])
    imgTF = tf.cast(imgTF, dtype=tf.float32)
    imgTF = (imgTF / 127.5) - 1.0
    imgTF = tf.expand_dims(imgTF, 0)

    # Get the generators
    gen_G = get_resnet_generator(name="generator_G")
    gen_F = get_resnet_generator(name="generator_F")

    # Get the discriminators
    disc_X = get_discriminator(name="discriminator_X")
    disc_Y = get_discriminator(name="discriminator_Y")

    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )

    # Load the checkpoints
    weight_file = "checkpoints/cyclegan_checkpoints.001"
    cycle_gan_model.load_weights(weight_file).expect_partial()
    print("Weights loaded successfully")
    prediction1 = cycle_gan_model.gen_G(imgTF, training=False)[0].numpy()
    prediction1 = (prediction1 * 127.5 + 127.5).astype(np.uint8)
    prediction1 = keras.preprocessing.image.array_to_img(prediction1)
    img1 = prediction1.resize((300, 300))
    img1 = ImageTk.PhotoImage(img1)

    weight_file = "checkpoints/cyclegan_checkpoints.020"
    cycle_gan_model.load_weights(weight_file).expect_partial()
    print("Weights loaded successfully")
    prediction2 = cycle_gan_model.gen_G(imgTF, training=False)[0].numpy()
    prediction2 = (prediction2 * 127.5 + 127.5).astype(np.uint8)
    prediction2 = keras.preprocessing.image.array_to_img(prediction2)
    img2 = prediction2.resize((300, 300))
    img2 = ImageTk.PhotoImage(img2)

    weight_file = "checkpoints/cyclegan_checkpoints.008"
    cycle_gan_model.load_weights(weight_file).expect_partial()
    print("Weights loaded successfully")
    prediction3 = cycle_gan_model.gen_G(imgTF, training=False)[0].numpy()
    prediction3 = (prediction3 * 127.5 + 127.5).astype(np.uint8)
    prediction3 = keras.preprocessing.image.array_to_img(prediction3)
    img3 = prediction3.resize((300, 300))
    img3 = ImageTk.PhotoImage(img3)

    weight_file = "checkpoints/cyclegan_checkpoints.012"
    cycle_gan_model.load_weights(weight_file).expect_partial()
    print("Weights loaded successfully")
    prediction4 = cycle_gan_model.gen_G(imgTF, training=False)[0].numpy()
    prediction4 = (prediction4 * 127.5 + 127.5).astype(np.uint8)
    prediction4 = keras.preprocessing.image.array_to_img(prediction4)
    img4 = prediction4.resize((300, 300))
    img4 = ImageTk.PhotoImage(img4)


    captureLabel = Label(captureFrame, image=img1)
    captureLabel.image = img1
    captureLabel.place(x=0, y=0)

    captureLabel = Label(captureFrame, image=img2)
    captureLabel.image = img2
    captureLabel.place(x=300, y=0)

    captureLabel = Label(captureFrame, image=img3)
    captureLabel.image = img3
    captureLabel.place(x=0, y=300)

    captureLabel = Label(captureFrame, image=img4)
    captureLabel.image = img4
    captureLabel.place(x=300, y=300)


    btnGenerate.destroy()
    btnSave = Button(btnFrame, text="SaveImage", image=imageBtSave, command=lambda: saveImage(prediction1, prediction2 ,prediction3 ,prediction4))
    btnSave.place(x=70, y=270)
    btnSave.config(bg="#000a01")


def showImage(path):
    global captureFrame
    global btnFrame
    global captureLabel
    global btnGenerate

    captureFrame = Frame()
    captureFrame.config(bg="white", width="600", height="600")  # white?
    captureFrame.place(x=30, y=0)

    img = Image.open(path)

    img = img.resize((600, 600), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)

    captureLabel = Label(captureFrame, image=img)
    captureLabel.image = img
    captureLabel.pack()

    btnFrame = Frame()
    btnFrame.config(bg="#000a01", width="394", height="600")
    btnFrame.place(x=630, y=0)

    btnGenerate = Button(btnFrame, text="GenerateImage", image=imageBtGenerate, command=lambda: generateImage(path))
    btnGenerate.place(x=10, y=270)
    btnGenerate.config(bg="#000a01")

    btnHome = Button(btnFrame, text="Home", image=imageBtHome, command=goHome)
    btnHome.place(x=340, y=545)


def uploadImage():
    file = filedialog.askopenfilename(title="Titulo de la ventana", filetypes=[("Archivos PNG", "*.png")])  # añadir initialdir= "C:/" si queremos que se abran en un directorio especifico
    if file != "":
        initFrame.pack_forget()

        # Cut it to make it square
        img = cv2.imread(file)
        altura, ancho, canales = img.shape
        pad = int((ancho - altura) / 2)
        img = img[0:int(altura), pad:pad + int(altura)]
        filename = 'uploadedImg.png'
        cv2.imwrite(filename, img)

        showImage(filename)


def takePhoto():
    global photo

    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

    # Cut it to make it square
    altura, ancho, canales = photo.shape
    pad = int((ancho-altura)/2)
    photo = photo[0:int(altura), pad:pad + int(altura)]

    cv2.imwrite('capturedImg.png', photo)  # Save image

    cameraObject.release()
    captureFrame.destroy()
    btnFrame.destroy()
    showImage('capturedImg.png')


def displayCamera():
    global photo

    if cameraObject is not None:
        retval, photo = cameraObject.read()
        if retval == True:
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)  # Change color to rgb
            photo = cv2.flip(photo, 1)      # Flip horizontally
            img = Image.fromarray(photo)
            img = img.resize((800, 600))  # 640x480 -> 800x600
            imgTk = ImageTk.PhotoImage(image=img)
            captureLabel.configure(image=imgTk)
            captureLabel.image = imgTk
            captureLabel.after(10, displayCamera)
        else:
            captureLabel.image = ""
            cameraObject.release()


def initCamera():
    global captureFrame
    global btnFrame
    global captureLabel

    global cameraObject

    # Load webcam
    messagebox.showinfo("Loading webcam...", "Wait while the webcam starts, it may take a few seconds.")
    cameraObject = cv2.VideoCapture(0)
    cameraObject.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cameraObject.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if cameraObject is not None:
        retval, photo = cameraObject.read()
        if retval == True:
            # Hide Init Frame
            initFrame.pack_forget()

            # Create Capture Frame
            captureFrame = Frame()
            captureFrame.config(width="600", height="600")
            captureFrame.pack()

            captureLabel = Label(captureFrame)
            captureLabel.config(width="600", height="600")
            captureLabel.pack()

            # Create Buttons Frame
            btnFrame = Frame()
            btnFrame.config(bg="#000a01", width="212", height="600")
            btnFrame.place(x=812, y=0)

            # Camera Button
            btnCamera = Button(btnFrame, text="Camera", image=imageBtCamera, command=takePhoto)
            btnCamera.place(x=30, y=270)

            # HomeButton
            btnHome = Button(btnFrame, text="Home", image=imageBtHome, command=goHome)
            btnHome.place(x=155, y=545)

            displayCamera()

        else:
            messagebox.showerror("Error", "Not webcam available")


def goHome():
    captureFrame.destroy()
    btnFrame.destroy()
    initFrame.pack()


def startApp():
    titleLabel.destroy()
    btnStart.destroy()

    # Take photo Button
    btnTakePhoto = Button(initFrame, text="TakePhoto", image=imageBtTakePhoto, command=initCamera)
    btnTakePhoto.place(x=110, y=180)
    btnTakePhoto.config(bg="#000a01")

    # Upload image Button
    btnUploadImage = Button(initFrame, text="UploadImage", image=imageBtUploadImage, command=uploadImage)
    btnUploadImage.place(x=98, y=360)
    btnUploadImage.config(bg="#000a01")


# ----------------------------------------------------------
# Root
root = Tk()
root.title("Face2Anime Translation")
root.geometry("1024x600")
root.resizable(False, False)
root.config(bg="#000a01")

# Init Frame
initFrame = Frame()
initFrame.config(bg="#000a01", width="1024", height="600")
initFrame.pack()

# Background image
imageBg = PhotoImage(file="imgs/ejemploFondo3.png")
bgLabel = Label(initFrame, image=imageBg, text="Background")
bgLabel.config(bg="#000a01")
bgLabel.place(x=60, y=0)

# Title image
imageTitle = PhotoImage(file="imgs/title.png")
titleLabel = Label(initFrame, image=imageTitle, text="Title")
titleLabel.place(x=10, y=160)
titleLabel.config(bg="#000a01")

# Init all de images for buttons
imageBtStart = PhotoImage(file="imgs/buttonStart.png")
imageBtTakePhoto = PhotoImage(file="imgs/buttonTakePhoto.png")
imageBtUploadImage = PhotoImage(file="imgs/buttonUploadImage.png")
imageBtCamera = PhotoImage(file="imgs/buttonCamera.png")
imageBtHome = PhotoImage(file="imgs/buttonHome.png")
imageBtGenerate = PhotoImage(file="imgs/buttonGenerateImage.png")
imageBtSave = PhotoImage(file="imgs/buttonSaveImage.png")

# Start Button
btnStart = Button(initFrame, text="Start", image=imageBtStart, command=startApp)
btnStart.place(x=130, y=380)
btnStart.config(bg="#000a01")


root.mainloop()
