/* feedreader.js
 *
 * This is the spec file that Jasmine will read and contains
 * all of the tests that will be run against your application.
 */

/* We're placing all of our tests within the $() function,
 * since some of these tests may require DOM elements. We want
 * to ensure they don't run until the DOM is ready.
 */
$(function() {
    /* This is our first test suite - a test suite just contains
    * a related set of tests. This suite is all about the RSS
    * feeds definitions, the allFeeds variable in our application.
    */
    describe('RSS Feeds', function() {
        /* This is our first test - it tests to make sure that the
         * allFeeds variable has been defined and that it is not
         * empty. Experiment with this before you get started on
         * the rest of this project. What happens when you change
         * allFeeds in app.js to be an empty array and refresh the
         * page?
         */
        it('are defined', function() {
            expect(allFeeds).toBeDefined();
            expect(allFeeds.length).not.toBe(0);
        });


        /* TODO: Write a test that loops through each feed
         * in the allFeeds object and ensures it has a URL defined
         * and that the URL is not empty.
         */
         it('each feed URL is defined and URL not empty', function () {
            for(var i = 0, len = allFeeds.length; i < len; i++) {
                expect(allFeeds[i].url).toBeDefined();
                expect(allFeeds[i].url.length).not.toBe(0);
            }
        });

        /* TODO: Write a test that loops through each feed
         * in the allFeeds object and ensures it has a name defined
         * and that the name is not empty.
         */
         it('each feed name is defined and name is not empty', function () {
            for(var i = 0, len = allFeeds.length; i < len; i++){
                expect(allFeeds[i].name).toBeDefined();
                expect(allFeeds[i].name.length).not.toBe(0);
            }
        });
    });


    /* TODO: Write a new test suite named "The menu" */
    describe('The menu', function () {

        var BODY = $('body'),
            MENU_BUTTON = $('.menu-icon-link');

        /* TODO: Write a test that ensures the menu element is
         * hidden by default. You'll have to analyze the HTML and
         * the CSS to determine how we're performing the
         * hiding/showing of the menu element.
         */
         it('menu element hidden by default', function () {
            expect(BODY.hasClass('menu-hidden')).toEqual(true);
        });

         /* TODO: Write a test that ensures the menu changes
          * visibility when the menu icon is clicked. This test
          * should have two expectations: does the menu display when
          * clicked and does it hide when clicked again.
          */
          it('menu changes visibility when clicked and hides the menu when click again', function () {
            MENU_BUTTON.click();
            expect(BODY.hasClass('menu-hidden')).toBe(false);

            MENU_BUTTON.click();
            expect(BODY.hasClass('menu-hidden')).toBe(true);
        });

      });

    /* TODO: Write a new test suite named "Initial Entries" */
    describe('Initial Entries', function(){
         beforeEach(function (done) {
             loadFeed(0, function () {
                 done();
             });

         });

        /* TODO: Write a test that ensures when the loadFeed
         * function is called and completes its work, there is at least
         * a single .entry element within the .feed container.
         * Remember, loadFeed() is asynchronous so this test will require
         * the use of Jasmine's beforeEach and asynchronous done() function.
         */
        it('at least a single .entry within .feed when loadFeed is called', function () {
            expect($('.feed').has('.entry').length).not.toBe(0);
        });
    });

    /* TODO: Write a new test suite named "New Feed Selection" */
    describe('New Feed Selection', function () {
        var newFeedSelectionHTML;

        beforeEach(function (done) {
            // load first feed
            loadFeed(0, function () {
                newFeedSelectionHTML = $('.feed').html();

                // Load second feed
                loadFeed(1, function () {
                    done();
                });
            });

        });
        

        /* TODO: Write a test that ensures when a new feed is loaded
         * by the loadFeed function that the content actually changes.
         * Remember, loadFeed() is asynchronous.
         */
        it('load new feed', function (done) {
            var newFeedHTML = $('.feed').html();
            expect(newFeedHTML).not.toBe(newFeedSelectionHTML);
            done();
        });

     });

}());
