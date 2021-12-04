Feed Reader Testing
===================
A web-based application that reads RSS feeds is provided and test suite is added using Jasmine.js with jQuery.

How to Run
----------
1. Open the folder and load 'index.html' in a web browser.
2. The feeds will load automatically

Test Suite
----------
1. RSS Feeds
- Test to make sure that the allFeeds variable has been defined and that it is not empty.
- Test loops through each feed in the allFeeds object and ensures it has a URL defined and that the URL is not empty.
- Test loops through each feed in the allFeeds object and ensures it has a name defined and that the name is not empty.

2. Menu
- Test to ensure the menu element is hidden by default.
- Test to ensure the menu changes visibility when the menu icon is clicked.

3. Initial Entries
- Test to ensure when the loadFeed function is called and completes its work, there is at least a single .entry element within the .feed container.

4. New Feed Selection
- Test to ensure when a new feed is loaded by the loadFeed function that the content actually changes.
