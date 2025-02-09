from pre.dataPipeLine import getProcesedData
from model.model import LipNetModel
from model.utils import CTCLoss
from tensorflow.keras.optimizers import Adam

data = getProcesedData("../data/s1/*.mpg")
train, test = data.take(2), data.skip(2).take(2)
frames, alignments = data.as_numpy_iterator().next()



model = LipNetModel()
model.compile(optimizer =Adam(), loss=CTCLoss)
model.fit(train, validation_data=test, epochs=10)