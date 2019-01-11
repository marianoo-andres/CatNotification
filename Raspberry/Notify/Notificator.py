import time
from SlackBot import SlackBot


class Notificator:
    PREDICTIONS_LENGTH = 6
    NOTIFICATION_DELAY = 60 #seconds

    def __init__(self):
        self.slackBot = SlackBot()
        self.predictions = [False for _ in range(Notificator.PREDICTIONS_LENGTH)]
        self.lastNotificationTime = time.time()

    def notify(self, imagePath):
        message = "Apareció un gato wachin!"
        channel = "#negra"
        self.slackBot.sendMessage(channel=channel, message=message)
        self.slackBot.sendImage(channel=channel, imagePath=imagePath)
    def manageNotification(self, prediction):
        label, prob = prediction
        labelTruthValue = True if label == 1 else False
        self.predictions.append(labelTruthValue)
        self.predictions.pop(0)
        count = 0
        lastPredIndex = None
        for i, pred in enumerate(self.predictions):
            if pred:
                lastPredIndex = i
                count += 1
        if count > Notificator.PREDICTIONS_LENGTH / 2:
            elapsed = time.time() - self.lastNotificationTime
            print("Elapsed {} seconds since last notification".format(elapsed))
            if elapsed > Notificator.NOTIFICATION_DELAY:
                print("Elapsed > NOTIFICATION_DELAY.. should notify")
                imageName = "temp{}.jpg".format(lastPredIndex + 1)
                self.notify(imageName)
                self.lastNotificationTime = time.time()