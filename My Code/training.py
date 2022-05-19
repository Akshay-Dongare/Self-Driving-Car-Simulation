print('Setting up...')
import utils
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# to remove warning msgs
utils.os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Step1
# Initialize the data
path = 'Generated Data from Virtual Cameras'
data = utils.importDataInfo(path)

# Step2
# Balance the data
utils.balanceData(data, display=0)  # display=1

# Step3
# Load the data
imagesPath, steerings = utils.loadData(path, data)

# Step4
# Split the data
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=10)  # random_state=5
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

# Step5
# Augment the data
# Step6
# Pre-process the data
# Step7
# Generate Batches


# Step8
# Create CNN with Keras Sequential API
model = utils.createModel()
model.summary()

# Step9 Train the model history = model.fit(batchGen(xTrain,yTrain,10,1),steps_per_epoch=20,epochs=2,
# validation_data=batchGen(xVal,yVal,10,0), validation_steps=20) history = model.fit(batchGen(xTrain, yTrain, 100,
# 1), steps_per_epoch=300, epochs=1000, validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200,
# callbacks=[monitor],verbose=2)
history = model.fit(utils.batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=1000,
              validation_data=utils.batchGen(xVal, yVal, 100, 0), validation_steps=200, callbacks=[utils.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5,
                            verbose=1, mode='auto', restore_best_weights=True)], verbose=2)
# Step10
# Save the model
model.save('heavydriver.h5')
print("Model Saved")

utils.plt.plot(history.history['loss'])
utils.plt.plot(history.history['val_loss'])
utils.plt.legend(["Training", "Validation"])
utils.plt.ylim(0, 1)
utils.plt.title('Loss')
utils.plt.xlabel('Epoch')
utils.plt.show()