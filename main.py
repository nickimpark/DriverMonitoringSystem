import cv2
from timeit import default_timer as timer
from datetime import datetime
import requests

from face_features_detector import FaceFeaturesDetector
from action_monitor import ActionMonitor
from condition_monitor import ConditionMonitor

if __name__ == '__main__':
    detector = FaceFeaturesDetector()
    action_monitor = ActionMonitor(detector)
    condition_monitor = ConditionMonitor(detector)
    cap = cv2.VideoCapture(0)

    # Handling flags
    isDistracted = 0
    Yawns = 0
    EyesClosed = 0
    IsSleeping = 0
    NoBlinking = 0
    IsUnconscious = 0

    while True:
        start = timer()
        _, frame = cap.read()
        detector.refresh(frame)

        try:
            frame = detector.annotated_frame()

            # Action monitor
            action_flags = action_monitor.refresh(frame)
            isDistractedCounter = action_flags[0]
            YawnsCounter = action_flags[1]

            frame = action_monitor.annotated_frame()
            # IsDistracted flag
            if action_monitor.isDistracted:
                isDistracted = 1
                cv2.putText(frame, "Водитель отвлечен!", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0, (50, 25, 150), 2)
            else:
                if isDistracted == 1:
                    print(f"{datetime.now()} Driver distracted for {round(isDistractedCounter, 2)} seconds")
                    requests.post("https://draconws.pythonanywhere.com/records",
                                  json=[{
                                      "type": "IsDistracted",
                                      "text": "Водитель отвлечен от дороги",
                                      "datetime": f"{datetime.now()}",
                                      "duration": isDistractedCounter
                                  }])
                    isDistracted = 0
            # Yawns flag
            if action_monitor.Yawns:
                Yawns = 1
                cv2.putText(frame, "Водитель зевает!", (50, 250), cv2.FONT_HERSHEY_COMPLEX, 1.0, (50, 25, 150), 2)
            else:
                if Yawns == 1:
                    print(f"{datetime.now()} Driver yawns for {round(YawnsCounter, 2)} seconds")
                    requests.post("https://draconws.pythonanywhere.com/records",
                                  json=[{
                                      "type": "Yawns",
                                      "text": "Водитель зевает",
                                      "datetime": f"{datetime.now()}",
                                      "duration": YawnsCounter
                                  }])
                    Yawns = 0

            # Condition monitor
            condition_flags = condition_monitor.refresh(frame)
            EyesClosedCounter = condition_flags[0]
            NoBlinkingCounter = condition_flags[1]

            frame = condition_monitor.annotated_frame()

            # EyesClosed & IsSleeping flag
            if condition_monitor.EyesClosed:
                EyesClosed = 1
                cv2.putText(frame, "Водитель засыпает!", (50, 300), cv2.FONT_HERSHEY_COMPLEX, 1.0, (50, 25, 150), 2)
                if condition_monitor.EyesClosedCounter > 5.0:
                    if not IsSleeping:
                        IsSleeping = 1
                        print(f"{datetime.now()} Driver is sleeping!")
                        requests.post("https://draconws.pythonanywhere.com/records",
                                      json=[{
                                          "type": "IsSleeping",
                                          "text": "Водитель уснул",
                                          "datetime": f"{datetime.now()}",
                                          "duration": None
                                      }])
            else:
                if EyesClosed == 1:
                    print(f"{datetime.now()} Driver's eyes closed for {round(EyesClosedCounter, 2)} seconds")
                    requests.post("https://draconws.pythonanywhere.com/records",
                                  json=[{
                                      "type": "EyesClosed",
                                      "text": "Водитель закрыл глаза",
                                      "datetime": f"{datetime.now()}",
                                      "duration": EyesClosedCounter
                                  }])
                    EyesClosed = 0
                    IsSleeping = 0

            # NoBlinking & IsUnconscious flag
            if condition_monitor.NoBlinking:
                NoBlinking = 1
                cv2.putText(frame, "Водитель слишком долго не моргает!", (50, 350), cv2.FONT_HERSHEY_COMPLEX, 1.0, (50, 25, 150), 2)
                if condition_monitor.NoBlinkingCounter > 40.0:
                    if not IsUnconscious:
                        IsUnconscious = 1
                        print(f"{datetime.now()} Driver is unconscious!")
                        requests.post("https://draconws.pythonanywhere.com/records",
                                      json=[{
                                          "type": "IsUnconscious",
                                          "text": "Водитель потерял сознание",
                                          "datetime": f"{datetime.now()}",
                                          "duration": None
                                      }])
            else:
                if NoBlinking == 1:
                    print(f"{datetime.now()} Driver doesn't blink for {round(NoBlinkingCounter, 2)} seconds")
                    requests.post("https://draconws.pythonanywhere.com/records",
                                  json=[{
                                      "type": "NoBlinking",
                                      "text": "Водитель не моргает",
                                      "datetime": f"{datetime.now()}",
                                      "duration": NoBlinkingCounter
                                  }])
                    NoBlinking = 0
                    IsUnconscious = 0

        except:
            pass

        cv2.imshow("Driver Monitoring System", frame)
        end = timer()
        fps = 1.0 / (end - start)
        action_monitor.update_fps(fps)
        condition_monitor.update_fps(fps)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
