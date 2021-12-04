# TechConf Registration Website

## Project Overview
The TechConf website allows attendees to register for an upcoming conference. Administrators can also view the list of attendees and notify all attendees via a personalized email message.

The application is currently working but the following pain points have triggered the need for migration to Azure:
 - The web application is not scalable to handle user load at peak
 - When the admin sends out notifications, it's currently taking a long time because it's looping through all attendees, resulting in some HTTP timeout exceptions
 - The current architecture is not cost-effective 

In this project, you are tasked to do the following:
- Migrate and deploy the pre-existing web app to an Azure App Service
- Migrate a PostgreSQL database backup to an Azure Postgres database instance
- Refactor the notification logic to an Azure Function via a service bus queue message

## Dependencies

You will need to install the following locally:
- [Postgres](https://www.postgresql.org/download/)
- [Visual Studio Code](https://code.visualstudio.com/download)
- [Azure Function tools V3](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=windows%2Ccsharp%2Cbash#install-the-azure-functions-core-tools)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)
- [Azure Tools for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode.vscode-node-azure-pack)

## Project Instructions

### Part 1: Create Azure Resources and Deploy Web App
1. Create a Resource group
2. Create an Azure Postgres Database single server
   - Add a new database `techconfdb`
   - Allow all IPs to connect to database server
   - Restore the database with the backup located in the data folder
3. Create a Service Bus resource with a `notificationqueue` that will be used to communicate between the web and the function
   - Open the web folder and update the following in the `config.py` file
      - `POSTGRES_URL`
      - `POSTGRES_USER`
      - `POSTGRES_PW`
      - `POSTGRES_DB`
      - `SERVICE_BUS_CONNECTION_STRING`
4. Create App Service plan
5. Create a storage account
6. Deploy the web app

### Part 2: Create and Publish Azure Function
1. Create an Azure Function in the `function` folder that is triggered by the service bus queue created in Part 1.

      **Note**: Skeleton code has been provided in the **README** file located in the `function` folder. You will need to copy/paste this code into the `__init.py__` file in the `function` folder.
      - The Azure Function should do the following:
         - Process the message which is the `notification_id`
         - Query the database using `psycopg2` library for the given notification to retrieve the subject and message
         - Query the database to retrieve a list of attendees (**email** and **first name**)
         - Loop through each attendee and send a personalized subject message
         - After the notification, update the notification status with the total number of attendees notified
2. Publish the Azure Function

### Part 3: Refactor `routes.py`
1. Refactor the post logic in `web/app/routes.py -> notification()` using servicebus `queue_client`:
   - The notification method on POST should save the notification object and queue the notification id for the function to pick it up
2. Re-deploy the web app to publish changes

## Monthly Cost Analysis
A monthly cost analysis (in USD) of each Azure resource to give an estimate total cost according to [Azure calculator](https://azure.microsoft.com/en-us/pricing/calculator/) in the table below:

| Azure Resource | Service Tier | Monthly Cost |
| ------------ | ------------ | ------------ |
| *Azure Postgres Database* | Single server/Basic Tier (Gen 5, 1 vCore, $24.82/month) x 1 selected + $0.10/GB/month x 5 GB selected + $0.10/GB/month x100 GB selected, Locally Redundant,  | $35.32 |
| *Azure Service Bus*   | Basic | $0.05/million messaging operation |
| *Azure App Service Plan* | Linux/Basic (B1: 1 Core, 1.75 GB RAM, 10 GB Storage, $13.14/month) x 1 selected | $13.14 |
| *Azure Storage acount* | Standard/Hot, StorageV2 (store $0.02/GB/month + write and list/create container $0.05/10,000 operations/month + read and other operations $0.004/10,000 operations/month) | $0.13 |
| *Azure Function App* | Consumption (first 400,000 GB/s of execution and 1 million executions free, beyond which $0.20/execution/month) | $0.00 |
| **Estimated Total Cost** | | **$48.64** |

## Architecture Explanation
The functions for the tech conference website, attendee registration and attendee notifications from confrence administrators via email, are simple enough and do not require much computing resources (< 14 GB RAM and < 4 vCPUs). As such, platform as a service (PaaS) like Azure Web App is ideal for this type of scenario. Setup is quick and easy, infrastructure is manaaged by Azure, and costs are lower than running on virtual machines. In the event more users are accessing the website, the Azure App Service Plan can scaled out (horizontal scaling) with more instances and vice versa. Additionally, if more computing resources are needed, scaling up (vertical scaling) is also managed within the App Service Plan.

Whenever the conference administrators send a new notification to all attendees from the website, the previous method of looping through all the attendees to send out individual emails was taking too long, causing HTPP timeouts. Moving this notification functionality to a background process with Azure Functions triggered by a service bus queue ensures all attendee email messages are sent out without failure. Additionally, since Azure Functions is an event-driven, serverless compute service that is triggered only on-demand, there is efficiency, scalability and cost effectiveness. 

With Azure Web App, being a PaaS, upfront infrastructure costs as well as costs for maintanence, are handled by Azure. Payment is only for the amount of resources used. For low traffic websites, there's a free tier which may be more than sufficient. With more traffic, scaling up and out are also managable within App Service Plan. The pay-as-you-go pricing also applies to Azure Functions, which charges for each execution beyond first million, and to Azure Service Bus, which charges for each message on per million basis. Overall for this application, the total costs are low.