neuronforest-spark
==================

(uses spark commit id: e895e0cbecbbec1b412ff21321e57826d2d0a982)

**To setup on EC2:**

./spark-ec2 -k luke -i ~/luke.pem -s 8 --instance-type=m3.medium --region=eu-west-1 --spark-version=e895e0cbecbbec1b412ff21321e57826d2d0a982 launch *NAME*

./spark-ec2 -k luke -i ~/luke.pem --region=eu-west-1 login *NAME*

curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"

unzip awscli-bundle.zip

sudo ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws

spark-ec2/copy-dir /usr/local/bin

*export aws credentials*

spark-ec2/copy-dir aws-credentials
