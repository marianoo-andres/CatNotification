import re
import time
from slackclient import SlackClient


class SlackBot:
    class Command:
        GREET = "saludar"

    # constants
    RTM_READ_DELAY = 1  # 1 second delay between reading from RTM
    EXAMPLE_COMMAND = "do"
    MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

    def __init__(self, slackBotToken=None):
        if not slackBotToken:
            slackBotToken = 'xoxb-382993025492-383119057715-0lIMr6v518JnNBcXbYYDewez'
        self.slackClient = SlackClient(slackBotToken)

        # Connect to api
        sucess = self.slackClient.rtm_connect(with_team_state=False)
        if not sucess:
            print("Connection failed. Exception traceback printed above.")
            return

        # Read bot's user ID by calling Web API method `auth.test`
        self.id = self.slackClient.api_call("auth.test")["user_id"]

        self.sendMessage(channel="#general", message="Hola gente estoy devuelta. Como va todo?")

    def start(self):
        """Start bot"""
        while True:
            slackEvents = self.slackClient.rtm_read()
            command, channel = self.__parseBotCommands(slackEvents)
            if command:
                self.handleCommand(command, channel)
            time.sleep(SlackBot.RTM_READ_DELAY)

    def __parseBotCommands(self, slackEvents):
        """
            Parses a list of events coming from the Slack RTM API to find bot commands.
            If a bot command is found, this function returns a tuple of command and channel.
            If its not found, then this function returns None, None.
        """
        for event in slackEvents:
            if event["type"] == "message" and not "subtype" in event:
                user_id, message = self.__parseDirectMention(event["text"])
                if user_id == self.id:
                    return message, event["channel"]
        return None, None

    def __parseDirectMention(self, messageText):
        """
            Finds a direct mention (a mention that is at the beginning) in message text
            and returns the user ID which was mentioned. If there is no direct mention, returns None
        """
        matches = re.search(SlackBot.MENTION_REGEX, messageText)
        # the first group contains the username, the second group contains the remaining message
        return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

    def sendMessage(self, channel, message):
        self.slackClient.api_call(
            "chat.postMessage",
            channel=channel,
            text=message
        )

    def sendImage(self, channel, imagePath, title='Status'):
        with open(imagePath, 'rb') as fileContent:
            self.slackClient.api_call(
                "files.upload",
                channels=channel,
                file=fileContent,
                title=title
            )

    def handleCommand(self, command, channel):
        """
        Executes bot command if the command is known
        """
        if command == SlackBot.Command.GREET:
            self.sendMessage(channel=channel, message="Holi soy Raspy jeje")
        else:
            self.sendMessage(channel=channel, message="Ni idea que me estás pidiendo che")
            self.sendImage(channel=channel, imagePath="confused.jpg", title="Ta re loco vo")
