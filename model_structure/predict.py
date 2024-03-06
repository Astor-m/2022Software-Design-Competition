from tkinter import Image
import numpy as np
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
 
model = load_model('model.h5')
#model.summary()#输出模型

predicate_image="1.jpg"

#将测试图片转为数组
im = Image.open(predicate_image)
im=im.resize((32,32))
im_L = im.convert("RGB")
Core = im_L.getdata()
arr1 = np.array(Core, dtype='float32') / 255.0

# arr1.shape
list_img = arr1.tolist()
#Z = list_img.reshape(1, 224, 224,3)
M=[]
M.extend(list_img)
Z = np.array(M).reshape(1, 32, 32,3)


x=['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
class_name = ['草履蚧', '麻皮蝽','丝带凤蝶','星天牛','桑天牛','松墨天牛','柳蓝叶甲','黄刺蛾','褐边绿刺蛾','霜天蛾']
dict_label = {1:'草履蚧', 2:'麻皮蝽',3:'丝带凤蝶',4:'星天牛',5:'桑天牛',6:'松墨天牛',7:'柳蓝叶甲',8:'黄刺蛾',9:'褐边绿刺蛾',10:'霜天蛾'}

class_name=['ActiasDubernardiOberthur', 'ActiasSeleneNingpoanaFelder', 'AgriusConvolvuli', 'AmsactaLactinea', 'AnoplophoraChinensisForster', 'AnoplophoraGlabripennisMotschulsky', 'AprionaGermari', 'AprionaSwainsoni', 'ArnpelophagaRubiginosaBremerEtGrey', 'AromiaBungiiFald', 'AtaturaIlia', 'BatoceraHorsfieldiHope', 'ByasaAlcinousKlug', 'CalospilosSuspectaWarren', 'CamptolomaInteriorata', 'CarposinaNiponensisWalsingham', 'CatharsiusMolossusLinnaeus', 'CeruraMencianaMoore', 'ChalcophoraJaponica', 'CicadellaViridis', 'ClanisBilineata', 'CletusPunctigerDallas', 'ClosteraAnachoreta', 'ClosteraAnastomosis', 'CnidocampaFlavescens', 'ConogethesPunctiferalis', 'CorythuchaCiliata', 'CreatonotusTransiens', 'CryptotympanaAtrataFabricius', 'CyclidiaSubstigmariaSubstigmaria', 'CyclopeltaObscura', 'CystidiaCouaggariaGuenee', 
'DanausChrysippusLinnaeus', 'DanausGenutia', 'DasychiraGroteiMoore', 'DendrolimusPunctatusWalker', 'DiaphaniaPerspectalis', 'DicranocephalusWallichi', 'DictyopharaSinica', 'DorcusTitanusPlatymelus', 'DrosichaCorpulenta', 'EligmaNarcissus', 'EnmonodiaVespertiliFabricius', 'ErthesinaFullo', 'EuricaniaClara', 'EurostusValidusDallas', 'EurydemaDominulus', 'GeishaDistinctissima', 'GraphiumSarpedonLinnaeue', 'GraphosomaRubrolineata', 'HalyomorphaPicusFabricius', 'HestinaAssimilis', 'HistiaRhodopeCramer', 'HyphantriaCunea', 'JacobiascaFormosana', 'LatoriaConsociaWalker', 'LethocerusDeyrolliVuillefroy', 'LocastraMuscosalisWalker', 'LycormaDelicatula', 'MegopisSinicaSinicaWhite', 'MeimunaMongolica', 'MicromelalophaTroglodyta', 'MiltochristaStriata', 'MonochamusAlternatusHope', 'Ophthalmitisirrorataria', 'OrthagaAchatina', 'PapilioBianorCramer', 'PapilioMachaonLinnaeus', 'PapilioPolytesLinnaeus', 'PapilioProtenorCramer', 'PapilioXuthusLinnaeus', 'ParocneriaFurva', 'PergesaElpenorlewisi', 'PidorusAtratusButter', 'PierisRapae', 'PlagioderaVersicolora', 'PlatypleuraKaempferi', 'PlinachtusBicoloripesScott', 'PlinachtusDissimilis', 'PolygoniaCaureum', 'PolyuraNarcaeaHewitson', 'PorthesiaSimilis', 'ProdeniaLitura', 'ProtaetiaBrevitarsisLewis', 'PsilogrammaMenephron', 'RicaniaSublimata', 'RiptortusPedestris', 'SemanotusBifasciatusBifasciatus', 'SericinusMontelusGrey', 'SinnaExtrema', 'SmerinthusPlanusWalker', 'SpeiredoniaRetorta', 'SpilarctiaRobusta', 'SpilarctiaSubcarnea', 'StilprotiaSalicis', 'TheretraJaponica', 'ThoseaSinensisWalker', 'UropyiaMeticulodina', 'VanessaIndicaHerbst']

pre = model.predict(Z)
print(pre)
print(np.argmax(pre))
print(class_name[int(np.argmax(pre))])#查看预测类型




   
