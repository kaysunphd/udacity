import React, { Component } from 'react';
import scriptLoader from 'react-async-script-loader';
import '../App.css';
import { MAP_API_KEY } from '../data/authorization';
import { FS_CLIENTID } from '../data/authorization';
import { FS_CLIENTSECRET } from '../data/authorization';
import { locations } from '../data/locations';
import { mapStyle } from '../data/map_style';
import markerUnSelected from '../img/red-pin.png';
import markerSelected from '../img/black-pin.png';
import foursquareLogo from '../img/foursquare.png';
import Map from './Map';
import Filter from './Filter';

let buildMap = {};
export let checkGetData = '';

class App extends Component {
  constructor(props) {
    super(props);

    //initialize
    this.state = {
      map: {},
      markers: [],
      infowindow: {}
    }
    this.getData = this.getData.bind(this);
  }

  componentWillReceiveProps({isScriptLoadSucceed}) {
    if (isScriptLoadSucceed) {
      this.getData();

      // initialize Google Maps
      buildMap = new window.google.maps.Map(document.getElementById('map'), {
        center: { lat: 33.7690, lng: -84.3880 },
        zoom: 12,
        styles: mapStyle,
        mapTypeControl: false,
        fullscreenControl: false
      });

      const buildInfoWindow = new window.google.maps.InfoWindow({maxWidth: 350});
      const bounds = new window.google.maps.LatLngBounds();
      const myEvents = 'click keypress'.split(' ');
      let buildMarkers = [];
      let allLocations = [];

      setTimeout(() => {
        //build markers from data information
        if (this.state.markers.length === 5) {
          allLocations = this.state.markers;
          checkGetData = true;

        } else {
          allLocations = locations;
          checkGetData = false;
        }

        for (let i = 0; i < allLocations.length; i++) {
          let position = {lat: allLocations[i].location.lat, lng: allLocations[i].location.lng};
          let name = allLocations[i].name;
          let address = allLocations[i].location.address;
          let lat = allLocations[i].location.lat;
          let lng = allLocations[i].location.lng;
          let bestPhoto = '';

          if (checkGetData === true) {
            bestPhoto = allLocations[i].bestPhoto.prefix.concat('width300', allLocations[i].bestPhoto.suffix);
          }

          let marker = new window.google.maps.Marker({
            id: i,
            map: buildMap,
            position: position,
            name: name,
            title: name,
            address: address,
            lat: lat,
            lng: lng,
            bestPhoto: bestPhoto,
            icon: markerUnSelected,
            animation: window.google.maps.Animation.DROP
          });

          buildMarkers.push(marker);

          // add event listeners to all created markers
          for (let i = 0; i < myEvents.length; i++) {
            marker.addListener(myEvents[i], function() {
              addInfoWindow(this, buildInfoWindow);
              this.setAnimation(window.google.maps.Animation.BOUNCE);
              setTimeout(function () {
                marker.setAnimation(null);
              }, 2000);
            });
          }

          marker.addListener('mouseover', function() {
            this.setIcon(markerSelected);
          });

          marker.addListener('mouseout', function() {
            this.setIcon(markerUnSelected);
          });

          bounds.extend(buildMarkers[i].position);
        }

        buildMap.fitBounds(bounds);

        // updates states with prepared data
        this.setState({
          map: buildMap,
          markers: buildMarkers,
          infowindow: buildInfoWindow
        });
      }, 800);

    // gmap error
    } else {
      alert('Google Maps failed to load, please refresh');
    }
  }

  //data from foursquare
  getData() {
    let places = [];
    locations.map((location) =>
      fetch(`https://api.foursquare.com/v2/venues/${location.venueId}` +
        `?client_id=${FS_CLIENTID}` +
        `&client_secret=${FS_CLIENTSECRET}` +
        `&v=20160922`)
        .then(response => response.json())
        .then(data => {
          if (data.meta.code === 200) {
            places.push(data.response.venue);
          }
        }).catch(error => {
          checkGetData = false;
          console.log(error);
        })
    );

    // updates the markers state with the data obtained
    this.setState({
      markers: places
    });
  }

  // Renders the App
  render() {
    return (
      <div className='App' role='main'>
        <Filter
          map={ this.state.map }
          markers={ this.state.markers }
          infowindow={ this.state.infowindow }
        />
        <Map />
      </div>
    );
  }
}

function addInfoWindow(marker, infowindow) {
  if (checkGetData === true) {
    infowindow.setContent(
      '<div class="info-wrap">'+
      '<h2 class="info-name">'+marker.name+'</h2><br>'+
      '<p class="info-position">Latitude: '+marker.lat+'</p>'+
      '<p class="info-position">Longitude: '+marker.lng+'</p><br>'+
      '<p class="info-address">Address: '+marker.address+'</p><br>'+
      '<img class="info-foursquare" src='+foursquareLogo+' alt="Powered by Foursquare"><br>'+
      '</div>'
    );
  } else {
    infowindow.setContent(
      '<div class="error-wrap">'+
      '<p class="error-message">Foursquare failed to load!</p><br>'+
      '</div>'
    );
  }
  infowindow.open(buildMap, marker);
}

export default scriptLoader(
  [`https://maps.googleapis.com/maps/api/js?key=${MAP_API_KEY}`]
)(App);
