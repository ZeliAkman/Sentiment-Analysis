#kütüphaneleri import etik

#hataları kapatma kodu
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from keras.datasets import imdb #imdb data setini indirdik internet baglantısına ihtiyacımız var keras yardımı ile indirecegiz
from keras.preprocessing.sequence import pad_sequences #veri setinin boyutu aynı olmak zorunda --> örnegin yorunların boyutlarını beli bir degere fixleyecegiz ona göre boyutu kısa olan yorumların başına ' 0 0 0 0 asdas 'ekleme yapacak
from keras.models import Sequential #Sequential bir model yaratacam gerekli olan tüm layer lerimi RNN, Embeding , dancelayer ,activation fonksiyonu kulanacam bunların hepsini Sequential yapıma ekliyorum yani bir dizi oluşturuyorum
from keras.layers.embeddings import Embedding #int leri belirli boyutlarda yogunluk vektorlerine çevirmemiz de yardımcı olacak bunu yaparkende beli başlı bir kelime sayısı kulanacagız ve okadar gelimeyi ind den dance vektorlere çevirecegiz 
from keras.layers import SimpleRNN, Dense, Activation #SimpleRNN-->bu bir rikorün nöral network layeri,Dense--> bildigimiz düz fletin layer buna sınıflandırma yapa bılmem ıcın ıhtıyacım var sornada sınıflandırma yapa bılmem ıcın bırtane de sigmoit fonksiyonu kulancam bunada --Activation-- layer içerisine ekleyecegim



(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path = "ibdb.npz",
                                                       num_words= None, #tüm kelimeleri çagırdık bu en çok kulanılak kelmimeleri sayılara atamıştık 
                                                       skip_top = 0, #en sık kulanılan kelimeleri ignor edip etmiyecegimizi yazıyoruz burada 0 suaun istemiyoruz
                                                       maxlen = None, #yorumlar 10 kelimelik ola bilir 100 kelimelik ola bilir bu parametre ile ilerde beliri sayıda yorum alacagız
                                                       seed = 113, #data üretilirken karıştırılıyor kerasın sayfasında da 113 dü aynı ortak sırayı bize veriyor demektir
                                                       start_char = 1, #yorumumuzdaki hangi karakterde başlayacagı bunu 1 yapın dıye kerasın içinde söylemiş
                                                       oov_char = 2, # bunun defualt degerı 2 bu bizimle ilgili bir parametre degil
                                                       index_from = 3) # bu da bizi ilgilendiren bir parametre de gil 

print("Type: ", type(X_train))
print("Type: ", type(Y_train))

print("X train shape: ",X_train.shape)
print("Y train shape: ",Y_train.shape)

#--> d = X_train[0] #0 ıncı ındekte bulunan yorumumuzun ıcınde 218 tane kelime vardır ve bu kelimelere karşılık gelen sayılar var 



# %% EDA

print("Y train values: ",np.unique(Y_train))#train ve test içerisinde kaçtane label var kaçtane class vaar hepsini görmemizi saglaycak , kaçtane uniq oalrak neler var onları görmemizi saglar
print("Y test values: ",np.unique(Y_test))

#ikitane clasımız var 0 ve 11 bunalrın dagılımına bakmak için şimdi 
#negatif görüşlerden 0:12500 ve pozitif görüşlerdende 1:12500 ve bu 100% dengeli bir veri seti
unique, counts = np.unique(Y_train, return_counts = True)
print("Y train distribution: ",dict(zip(unique,counts)))

unique, counts = np.unique(Y_test, return_counts = True)
print("Y testdistribution: ",dict(zip(unique,counts)))

#countplot kulanarak bu verileri görselleştirdik
plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y train")

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y test")

d = X_train[0]
print(d)
print(len(d)) #toplamda 218 tane kelime var 

review_len_train = [] #kaç kelime var onlara bakmak için boş listeler açtık
review_len_test = []
for i, ii in zip(X_train, X_test):
    review_len_train.append(len(i)) #yorumlardaki kelime sayılarını bu açtıgımız boş listelere ekledik ve saılarına baktık
    review_len_test.append(len(ii))
    
#insnaların yaptıkları yorumlardaki dagılıma bakalım şimdi 
#bunun için siborn kütüpahnesini kulanıyoruz
sns.distplot(review_len_train, hist_kws = {"alpha":0.3}) #alpha saydamlık
sns.distplot(review_len_test, hist_kws = {"alpha":0.3})

#Gırafige baktıgımızda kuyruk sag dogru uzamış yani burda pozitif secuiruns görüyoruz normal dagılıma sahıpdegil biraz sola yatmış bir dagılıma sahip
#median ve mean degerelerine bakacagız ..
print("Train mean:", np.mean(review_len_train)) #dagılımımızın orta noktası
print("Train median:", np.median(review_len_train)) #
print("Train mode:", stats.mode(review_len_train)) #En tepe noktası

# number of words(Kaç tane kelime var ona baklım)
word_index = imdb.get_word_index() # kelimelerin indexlerini alıp bunu word indexe eşitledik
print(type(word_index)) #typlarına baktık
print(len(word_index))

for keys, values in word_index.items(): # burda 22 ci indexte hangi kelime kulanılmış ona baktırk
    if values == 22:                    # you ya karşılık geliyor
        print(keys)

def whatItSay(index = 24):   #defualt olarak 24 degerini alsın
    #
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i - 3, "!") for i in X_train[index]]) # ! ekledik
    print(decode_review) 
    print(Y_train[index]) #görüşn olumlumu olumsumu olduguna bakacagız
    return decode_review

decoded_review = whatItSay(36) #36 cı yoruma baktık burda bir kaç farklı degere bak ve yorumları gör nasıl atadgımızı içine ne yazdıgımıza bak

# %% Preprocess
# veriyi train edile bilir hale getirmek için yapacagımız iktane işlem var
num_words = 15000  #Kelime sınırımızı belirledik
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words) #bu 15000 lik parametriyi kulana bilmek için veri setimi tekrardan çagıracagız

maxlen = 130 # yorum uzunlugu 
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print(X_train[5]) #5.yoruma baktık kısa oldugu için başına 0 yada ! ekledik ekledi ve 130 a Tamamladı

for i in X_train[0:10]:
    print(len(i))

decoded_review = whatItSay(5)


# %% RNN

rnn = Sequential() #bir dizimiz olacak bunun içine layerleri ekleyecegiz
rnn.add(Embedding(num_words, 32, input_length = len(X_train[0])))
rnn.add(SimpleRNN(16, input_shape = (num_words,maxlen), return_sequences= False, activation= "relu"))
rnn.add(Dense(1)) #sınıflandırma işlemini gerçekleştire bilmek için bunu danse layer ile yapıyoruz
rnn.add(Activation("sigmoid")) #Rnn yapımıza Activation eklememiz gerekiyor sınıfladnırma işlemini gerçekleştire

print(rnn.summary()) #layerlerimizin ve parametrilerimizin sayısını gösterecek
rnn.compile(loss = "binary_crossentropy", optimizer="rmsprop",metrics= ["accuracy"])

# #-->Egitim ve Test verilerini oluşturduk ve egitik
history = rnn.fit(X_train, Y_train, validation_data= (X_test, Y_test), epochs=5, batch_size= 128, verbose=1)
#test scorlarımızda bizim en başarılı oldugumuz  4 epoch da elde etigimiz degerlermiş 

#test scorumuzu ekrana yazdırdık
score = rnn.evaluate(X_test, Y_test)
print("Accuracy: %",score[1]*100)

#los ve Accuracy degerlerinin nasıl degiştigini birde plot üzerinden görerleim
plt.figure()
plt.plot(history.history["accuracy"], label = "Train")
plt.plot(history.history["val_accuracy"], label = "Test")
plt.title("val_acc")
plt.ylabel("val_acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()
#gıragiklere baktıgımızda en iyi degerimizi birinci epoch da üretmişiz


plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()









































