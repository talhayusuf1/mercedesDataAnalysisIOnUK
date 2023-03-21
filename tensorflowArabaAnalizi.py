import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential  # MOdeli olsuturuyruz
from keras.layers import Dense  # katrmanları olusturuyoruz
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Bir data ile ugrasırken ilk yapoacagımız is veriyi okumak herhangi bir
# eksik olup olmadıgını kontrol ederiz  null varsa ya dolduruuz yada sileriz


# *************************VERİYİ OKUMA VE ANLAMA*****************************
dataFrame = pd.read_excel("merc.xlsx")  # Veriyi okuruz
print(dataFrame.head())  # ilk 5 veriyi getirir
print(dataFrame.describe())  # Verinin ufak caplı analizini yapar
print(dataFrame.isnull().sum())  # Hangi kolonda kac tane veride eksik var

sbn.distplot(dataFrame["price"])  # Dagilim grafigi cizer
plt.show()
sbn.countplot(dataFrame["year"])
plt.show()
print(dataFrame.corr())  # Verilerin birbirleri arasındaki korelasyon
print(dataFrame.corr()["price"].sort_values())
# Mesela bu veride -olarak gozuklenler fiyatı dusuruyor + olanlar ise fiyatı
# yukselten etkenlerdir
# Noktasal grafik iki deger giriyoruz
sbn.scatterplot(x="mileage", y="price", data=dataFrame)
plt.show()
# ************************VERİYİ TEMİZLEME************************************
# Ascending false en yuksek fiyatı en yukarda getirir
print(dataFrame.sort_values("price", ascending=False).head(20))

print(dataFrame.sort_values("price", ascending=True).head(20))
# BUrda yapmaya calıstıgım sey yuzde 1lik kısmını veriden cıkarmak
# cunku veride tutarsızlık var
print(len(dataFrame) * 0.01)
# Burda 131 veriyi sildikk
yeniDataFrame = dataFrame.sort_values("price", ascending=False).iloc[131:]
print(yeniDataFrame.describe())
plt.figure(figsize=(7, 5))
sbn.distplot(yeniDataFrame["price"])
plt.show()

print(dataFrame.describe())
# Yillara gore fiyat ortalamasını aldık cunku fiyatta bir abzurtluk var mı yokmu ona bakıyoruz
print(dataFrame.groupby("year").mean()["price"])
# 1970 de bi sacmalıok oldugunu gırduk

print(dataFrame[dataFrame.year != 1970].groupby("year").mean()["price"])

dataFrame = yeniDataFrame

print(dataFrame.describe())

dataFrame = dataFrame[dataFrame.year != 1970]
print(dataFrame.groupby("year").mean()["price"])

dataFrame = dataFrame.drop("transmission", axis=1)

# ************************VERİYİ TRAİN TEST OLARAK AYIRMAK***********************
y = dataFrame["price"].values
x = dataFrame.drop("price", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=10)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)  # burda kac tane features oldugunu gorduk

model = Sequential()
# Bunu deniyoruz
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))  # katmanlar
model.add(Dense(12, activation="relu"))

model.add(Dense(1))  # Cıkıs katmanı 1 tane yeterli
model.compile(optimizer="adam", loss="mse")
# batch_size veriyi parcalara boler
model.fit(x=x_train, y=y_train, validation_data=(
    x_test, y_test), batch_size=250, epochs=300)

kayipVerisi = pd.DataFrame(model.history.history)
# Burdaki veride birbirinden ayrı cıksaydı epochu yada baska bir seyi degistirmek zorundaydık
print(kayipVerisi)
kayipVerisi.plot()
plt.show()

tahminDizisi = model.predict(x_test)

print(tahminDizisi)

# burda nekadarlik bir fark oldugunu bulmaua calıstık
mean_absolute_error(y_test, tahminDizisi)

plt.scatter(y_test, tahminDizisi)
plt.plot(y_test, y_test, "g*-")
plt.show()

dataFrame.iloc[2]

yeniArabaSeries = dataFrame.drop("price", axis=1).iloc[2]

yeniArabaSeries = scaler.transform(yeniArabaSeries.values.reshape(-1, 5))

print(model.predict(yeniArabaSeries))
