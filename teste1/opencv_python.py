import cv2

webCamera = cv2.VideoCapture(2)
classificadorVideoFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #XML com algoritmo de aprendizado de máquina

while True:
    camera, frame = webCamera.read()#abre a webcam

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecta = classificadorVideoFace.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=8,  minSize=(25, 25)) #parametros de precisão, se ajusta a ter um nivel satisfatorio 
    for(x, y, l, a) in detecta:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2) #detecção da face

        contador = str(detecta.shape[0])#Adiciona as faces reconhecidas em um contador

        cv2.putText(frame, contador, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) #Texto da qunatidades e retangulo para ficar mais visivel o reconhecimento

        cv2.putText(frame, "Quantidade de Faces: " + contador, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) #Texto ilustrando a quantidade reconehcida de rosto 


    cv2.imshow("Video WebCamera", frame)#Exibir a imagem da webcam

    if cv2.waitKey(1) == ord('f'):
        break

webCamera.release()
cv2.destroyAllWindows()