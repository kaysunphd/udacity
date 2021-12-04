import logging
import azure.functions as func
import psycopg2
import os
from datetime import datetime
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def main(msg: func.ServiceBusMessage):

    notification_id = int(msg.get_body().decode('utf-8'))
    logging.info('Python ServiceBus queue trigger processed message: %s',notification_id)

    # Get connection to database
    connection = psycopg2.connect(host="*****",
                            database="*****",
                            user="*****",
                            password="*****")
    cursor = connection.cursor()

    try:
        # Get notification message and subject from database using the notification_id
        cursor.execute('SELECT message, subject FROM notification WHERE id=%s;', (notification_id,))
        notify_message, notify_subject = cursor.fetchone()
        logging.info('Error with Notification ID: {}, Message: {}, Subject: {}'.format(notification_id, notify_message, notify_subject))

        # Get attendees email and name
        cursor.execute('SELECT email, first_name FROM attendee;')
        attendees = cursor.fetchall()

        # Loop through each attendee and send an email with a personalized subject
        # from https://app.sendgrid.com/guide/integrate/langs/python
        for (email, first_name) in attendees:
            message = Mail(
                from_email='nysuka@gmail.com',
                to_emails=email,
                subject='{}: {}'.format(first_name, notify_subject),
                html_content='hi {}, {}'.format(first_name, notify_message))
            try:
                sg = SendGridAPIClient('******')
                response = sg.send(message)
            except Exception as e:
                logging.error(e.message)

        # Update the notification table by setting the completed date and updating the status with the total number of attendees notified
        complete_date = datetime.utcnow()
        status = 'Notified {} number of attendees.'.format(len(attendees))
        cursor.execute('UPDATE notification SET status=%s, completed_date=%s WHERE id=%s;', (status, complete_date, notification_id))
        connection.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(error)

    finally:
        # Close connection
        if connection is not None:
            cursor.close()
            connection.close()