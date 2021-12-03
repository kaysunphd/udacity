import pandas as pd
import boto3
import json
import sys


def load_configuration():
    """
    Load configuration parameters to AWS.
    """
    import configparser
    config = configparser.ConfigParser()
    config.read_file(open('dwh.cfg'))
    config_dict = dict()

    config_dict['KEY']                    = config.get('AWS','KEY')
    config_dict['SECRET']                 = config.get('AWS','SECRET')

    config_dict['CLUSTER_TYPE']       = config.get("DWH","CLUSTER_TYPE")
    config_dict['NUM_NODES']          = config.get("DWH","NUM_NODES")
    config_dict['NODE_TYPE']          = config.get("DWH","NODE_TYPE")
    config_dict['IAM_ROLE_NAME']      = config.get("DWH", "IAM_ROLE_NAME")
    config_dict['CLUSTER_IDENTIFIER'] = config.get("DWH","CLUSTER_IDENTIFIER")
    
    config_dict['DB_NAME']                 = config.get("CLUSTER","DB_NAME")
    config_dict['DB_USER']            = config.get("CLUSTER","DB_USER")
    config_dict['DB_PASSWORD']        = config.get("CLUSTER","DB_PASSWORD")
    config_dict['DB_PORT']               = config.get("CLUSTER","DB_PORT")
    
    config_dict['POLICY']               = config.get("IAM_ROLE", "POLICY")
    config_dict['ARN']               = config.get("IAM_ROLE", "ARN")

    return config_dict



def create_clients(config_dict):
    """
    Create clients for EC2, S3, IAM and Redshift
    """
    ec2 = boto3.resource('ec2',
                         region_name="us-east-1",
                         aws_access_key_id=config_dict['KEY'],
                         aws_secret_access_key=config_dict['SECRET']
                        )

    s3 = boto3.resource('s3',
                         region_name="us-east-1",
                         aws_access_key_id=config_dict['KEY'],
                         aws_secret_access_key=config_dict['SECRET']
                       )

    iam = boto3.client('iam',
                        aws_access_key_id=config_dict['KEY'],
                        aws_secret_access_key=config_dict['SECRET'],
                        region_name='us-east-1'
                      )

    redshift = boto3.client('redshift',
                             region_name="us-east-1",
                             aws_access_key_id=config_dict['KEY'],
                             aws_secret_access_key=config_dict['SECRET']
                           )
    
    print('Success creating clients.')
    
    return ec2, s3, iam, redshift

    
def create_iam_role(config_dict, iam):
    """
    Create iam role.
    """    
    try:
        print('Creating a new IAM Role')
        dwhRole = iam.create_role(Path='/',
                                  RoleName=config_dict['IAM_ROLE_NAME'],
                                  Description = "Allows Redshift clusters to call AWS services on your behalf.",
                                  AssumeRolePolicyDocument=json.dumps(
                                      {'Statement': [{'Action': 'sts:AssumeRole',
                                                      'Effect': 'Allow',
                                                      'Principal': {'Service': 'redshift.amazonaws.com'}}],
                                       'Version': '2012-10-17'})
                                  )    
    except Exception as e:
        print(e)
        
    try:
        print('Creating role policy')
        iam.attach_role_policy(RoleName=config_dict['IAM_ROLE_NAME'],
                               PolicyArn=config_dict['ARN']
                               )['ResponseMetadata']['HTTPStatusCode']
    except Exception as e:
        print(e)
        
    roleArn = iam.get_role(RoleName=config_dict['IAM_ROLE_NAME'])['Role']['Arn']
    print('role arn: ', roleArn)
    
    return roleArn
    

def open_tcp_port(myClusterProps):
    try:
        vpc = ec2.Vpc(id=myClusterProps['VpcId'])
        defaultSg = list(vpc.security_groups.all())[-1]
        print(defaultSg)

        defaultSg.authorize_ingress(
            GroupName= defaultSg.group_name,
            CidrIp='0.0.0.0/0',
            IpProtocol='TCP',
            FromPort=int(config_dict['DB_PORT']),
            ToPort=int(config_dict['DB_PORT'])
        )
    except Exception as e:
        print(e)
    
    
def create_cluster(config_dict, roleArn, redshift):
    try:
        print('Creating redshift cluster')
        response = redshift.create_cluster(ClusterType=config_dict['CLUSTER_TYPE'],
                                           NodeType=config_dict['NODE_TYPE'],
                                           NumberOfNodes=int(config_dict['NUM_NODES']),
                                           DBName=config_dict['DB_NAME'],
                                           ClusterIdentifier=config_dict['CLUSTER_IDENTIFIER'],
                                           MasterUsername=config_dict['DB_USER'],
                                           MasterUserPassword=config_dict['DB_PASSWORD'],
                                           IamRoles=[roleArn] 
                                           )
        
        myClusterProps = redshift.describe_clusters(ClusterIdentifier=config_dict['CLUSTER_IDENTIFIER'])['Clusters'][0]
        if myClusterProps['ClusterStatus'] == 'available':
            open_tcp_port(myClusterProps)
            
    except Exception as e:
        print(e)
    
    
def prettyRedshiftProps(props):
    pd.set_option('display.max_colwidth', -1)
    keysToShow = ["ClusterIdentifier", "NodeType", "ClusterStatus", "MasterUsername", "DBName", "Endpoint", "NumberOfNodes", 'VpcId']
    x = [(k, v) for k,v in props.items() if k in keysToShow]
    return pd.DataFrame(data=x, columns=["Key", "Value"])


def start_cluster():
    config_dict = load_configuration()
    ec2, s3, iam, redshift = create_clients(config_dict)
    roleArn = create_iam_role(config_dict, iam)
    create_cluster(config_dict, roleArn, redshift)
    

def stop_cluster():
    config_dict = load_configuration()
    iam = boto3.client('iam',
                        aws_access_key_id=config_dict['KEY'],
                        aws_secret_access_key=config_dict['SECRET'],
                        region_name='us-east-1'
                      )
    
    redshift = boto3.client('redshift',
                             region_name="us-east-1",
                             aws_access_key_id=config_dict['KEY'],
                             aws_secret_access_key=config_dict['SECRET']
                           )
    
    redshift.delete_cluster( ClusterIdentifier=config_dict['CLUSTER_IDENTIFIER'],  SkipFinalClusterSnapshot=True)
    iam.detach_role_policy(RoleName=config_dict['IAM_ROLE_NAME'], PolicyArn=config_dict['ARN'])
    iam.delete_role(RoleName=config_dict['IAM_ROLE_NAME'])
    
    
def main():
    if sys.argv[1] == 'start':
        start_cluster()
    elif sys.argv[1] == 'stop':
        stop_cluster()
    else:
        print('Use either start or stop.')
        
    
if __name__ == "__main__":
    main()