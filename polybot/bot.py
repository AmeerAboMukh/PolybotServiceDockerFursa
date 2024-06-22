import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3
import requests
class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60,
                                             certificate=open('/home/ubuntu/YOURPUBLIC.pem', 'r'))

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text, timeout=5):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id,
                                              timeout=5)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path),
            timeout = 5
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')

#  upload the photo to S3
#  send an HTTP request to the `yolo5` service for prediction
#  send the returned results to the Telegram end-user
class ObjectDetectionBot(Bot):
    def __init__(self, token, telegram_chat_url, s3_bucket_name, yolo5_service_url):
        super().__init__(token, telegram_chat_url)
        self.s3_bucket_name = s3_bucket_name
        self.yolo5_service_url = yolo5_service_url
        self.s3_client = boto3.client('s3')

    def upload_to_s3(self, file_path):
        file_name = os.path.basename(file_path)
        self.s3_client.upload_file(file_path, self.s3_bucket_name, file_name)
        return file_name

    def get_predictions(self, image_url):
        response = requests.post(self.yolo5_service_url, json={"image_url": image_url})

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)
            self.send_text(msg['chat']['id'], 'I received. Starting the process...')

            try:
                s3_url = self.upload_to_s3(photo_path)
                logger.info(f'The photo uploaded to s3')

                predictions = self.get_predictions(s3_url)
                logger.info('Predictions received')

            except Exception as e:
                logger.error(f'Error occur while handling the message:  {e}')
                self.send_text(msg['chat']['id'],f'Error!!: {e}')
        else:
            self.send_text(msg['chat']['id'], 'Please send a photo for object detection.')
