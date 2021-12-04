/*
 * Create a list that holds all of your cards
 */
let cards = ['diamond', 'paper-plane', 'anchor', 'bolt', 'cube', 'leaf', 'bicycle', 'bomb',
			'diamond', 'paper-plane', 'anchor', 'bolt', 'cube', 'leaf', 'bicycle', 'bomb'];

let pickedCards = [];
let numOfMoves;
let numOfMatches;
let startTime;
let seconds;

/*
 * Display the cards on the page
 *   - shuffle the list of cards using the provided "shuffle" method below
 *   - loop through each card and create its HTML
 *   - add each card's HTML to the page
 */

// initialize game
function initialize() {
	//clear deck
	$('.deck').empty();

	// reset count
	$('.moves').text('0');
	numOfMoves = 0;
	numOfMatches = 0;

	// reset stars
	$('.fa-star').removeClass('fa-star-o').addClass('fa-star');
	$('.fa-star-o').removeClass('fa-star-o').addClass('fa-star');

	// shuffle
	let shuffledCards = shuffle(cards);

	// Create grid
	for (let row = 0; row < shuffledCards.length; row++) {
		let makeRow = $('<li class="card"><i class="fa fa-' + shuffledCards[row] + '"></i></li>');
		$('.deck').append(makeRow);
 	}

 	onClick();

 	// setup timer
 	seconds = 0;
 	$('.timer').text(seconds);
	clearInterval(startTime);
 	onTimer();
 }

// stopwatch
function onTimer() {
    startTime = setInterval(function () {
            $('.timer').text(seconds);
            ++seconds;
        }, 1000); // equals 1 second
}

// Shuffle function from http://stackoverflow.com/a/2450976
function shuffle(array) {
    var currentIndex = array.length, temporaryValue, randomIndex;

    while (currentIndex !== 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
    }

    return array;
}

// win game
function wining(moveNum, starNum) {
	$('.modal').modal('show');

	starScore(moveNum);
	starNum = $('.score-panel').find('.fa-star').length;

	document.getElementById('numMoves').innerHTML = moveNum;
    document.getElementById('numStars').innerHTML = starNum;
    document.getElementById('duration').innerHTML = seconds;

    $('.replay').on('click', function () {
            $('.modal').removeData('bs.modal');
            $('.modal').modal('hide');
            initialize();
        });
}

// star score
function starScore(moveNum) {
	var twinkles = $('.stars').children().find('.fa');

    if (moveNum === 12 && twinkles[2].classList.contains('fa-star')) {
        twinkles[2].classList.toggle('fa-star');
        twinkles[2].classList.toggle('fa-star-o');
    }
    else if (moveNum === 24 && twinkles[1].classList.contains('fa-star')) {
        twinkles[1].classList.toggle('fa-star');
        twinkles[1].classList.toggle('fa-star-o');
    }
	else if (moveNum === 30 && twinkles[0].classList.contains('fa-star')) {
        twinkles[0].classList.toggle('fa-star');
        twinkles[0].classList.toggle('fa-star-o');
    }

}

// on click
function onClick() {
	$('.deck').find('.card').on('click', function () {

		if ($(this).hasClass('show') || $(this).hasClass('match')) {
			return;
		}

		//flip card
        $(this).addClass('open show');

        // check if match
        pickedCards.push($(this).html());

        if (pickedCards.length === 2) {
        	// matches
        	if (pickedCards[0] === pickedCards[1]) {
        		$('.deck').find('.open').addClass('open show match');
        		numOfMatches += 2;
        	}
        	// dont match
        	else {
        		// need a delay to close both cards
				setTimeout(function () {
					$('.deck').find('.open').removeClass('open show');
				}, 400);
        	}

        	pickedCards = [];

        	//update moves counter
       		$('.moves').html(++numOfMoves);
       		starScore(numOfMoves);

        }

        // if all matches
        if (numOfMatches == cards.length) {
        	wining(numOfMoves);
        }

	});
}

//reset
$('.restart').on('click', function () {
    initialize();
});

// start game
initialize();
