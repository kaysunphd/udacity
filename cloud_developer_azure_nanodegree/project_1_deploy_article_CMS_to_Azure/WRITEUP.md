# Write-up

### Analyze, choose, and justify the appropriate resource option for deploying the app.

*For **both** a VM or App Service solution for the CMS app:*
- *Analyze costs, scalability, availability, and workflow*
- *Choose the appropriate solution (VM or App Service) for deploying the app*
- *Justify your choice*
---
**App Service:**
Since this CMS is quite small, not requiring any significant computation, memory or other resources, the costs are low for small apps and with PaaS, maintenance support is minimum. Depending on number of users, which can increase or decrease with time, App Service allows elasticity that can be handled automatically and price scaled down with low usage. There are data centers around the globe for low latency to users and high availability, with additional optional support for disaster recovery. Deployment with App Service is straightforward with automatic CI/CD using code repositories like GitHub or Azure or private repo.

**VM:**
With the small app requiring minimum resources, the cost to host the VM is also relatively low. However, maintenance support to ensure up-to-date OS and dependency software will be ongoing. Support staff will have full access and control of the VM for greater customizations, particularly with custom images. To ensure sufficient VMs are created to support the fluctuating number of users, scaling can be achieved with Virtual Machine Scale Sets and Load Balancers. Availability is also high with data centers all around the world to ensure low latency. Deployment is more hands more and manual.

**Summary of App Services vs. VM for cost, scalability, and availability.**

|                       | App Service                                                                     | VM                                                    |
|-----------------------|---------------------------------------------------------------------------------|-------------------------------------------------------|
| minimum hardware      | 1 Core/ 1.75 GB RAM/ 10 GB storage/ Basic tier/ Linux                           | 1 Core/ 1.75 GB RAM/ 40 GB storage/ Basic tier/ Linux |
| minimum monthly cost  | $12.41                                                                          | $16.84                                                 |
| autoscaling           | built-in                                                                        | VM scale set                                          |
| load balancer         | integrated                                                                      | Azure Load Balancer                                   |
| scale limit           | 30 instances, 100 with App Service Environment                                  | platform image: 1000 nodes per scale set/ custom image: 600 nodes per scale set|
| SLA                   | 99.95%                                                                          | 95% to 99.95%                                         |
| multi-region failover | traffic manager                                                                 | traffic manager                                       |
Source:
- https://docs.microsoft.com/en-us/azure/architecture/guide/technology-choices/compute-decision-tree#scalability
- https://azure.microsoft.com/en-in/pricing/calculator/

**Decision:**
The small size of the app and computing resources required leads to App Service as the preferred solution. Low cost, highly scalable, high availability and straightforward workflow are all achievable with App Service. Even though VM would result in similar monthly costs, high scalability and high availability, it would be overkill for this small app, particularly for the higher costs for personnel to maintain, deploy and upgrade. 

### Assess app changes that would change your decision.

*Detail how the app and any other needs would have to change for you to change your decision in the last section.*

Since App Service is limited to maximum of 4 vCPUs and 14 GB of RAM, a more computational intensive app that challenges this hardware limitation, such as with user interactivity, videos, etc., will warrant the shift to VM, where there are much larger computing resources. The more complex app will also mean large code base, making a more direct deployment much faster than in App Service. Added complexity may also come in the form of modifications to OS and/or dependency software and libraries, which will require installation of custom OS images that can only be achieved with VMs.     