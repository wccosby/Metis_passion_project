import os
from flask import Flask, request, Response
from slackclient import SlackClient
from twilio import twiml
from twilio.rest import TwilioRestClient

TWILIO_NUMBER = os.environ.get('TWILIO_NUMBER', None)
USER_NUMBER = os.environ.get('USER_NUMBER', None)

app = Flask(__name__)
slack_client = SlackClient(os.environ.get('SLACK_TOKEN', None))
twilio_client = TwilioRestClient()

@app.route('/twilio', methods=['POST'])
def twilio_post():
    response = twiml.Response()
    if request.form['From'] == USER_NUMBER:

        response.message("this could be the neural network output!")
        message = request.form['Body']
        # put the message in slack
        slack_client.api_call("chat.postMessage", channel='#general',
                                text=message, username='stratusbot',
                                icon_emoji=':robot_face:')

        return str(response) # returns a message to the user
        # return Response(response.toxml(), mimetype="text/xml"), 200, str(message)

if __name__ == '__main__':
    app.run(debug=True)
