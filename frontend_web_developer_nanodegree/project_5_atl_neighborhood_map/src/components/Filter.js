import React, { Component } from 'react';
import markerUnSelected from '../img/red-pin.png';
import markerSelected from '../img/black-pin.png';
import foursquareLogo from '../img/foursquare.png';
import { checkGetData } from './App';

class Filter extends Component {
  // Constructor
  constructor(props) {
    super(props);

    // initalize
    this.state = {
      query: '',
      map: {},
      markers: [],
      infowindow: {},
      currentMarkers: []
    }

    this.showFilter = this.showFilter.bind(this);
    this.hideFilter = this.hideFilter.bind(this);
    this.markerFilter = this.markerFilter.bind(this);
    this.openInfoWindow = this.openInfoWindow.bind(this);
  }

  componentWillMount() {
    setTimeout(() => {

      this.setState({
        map: this.props.map,
        markers: this.props.markers,
        infowindow: this.props.infowindow,
        currentMarkers: this.props.markers
      });
    }, 1000);
  }

  //eopn filter
  showFilter() {
    const filter = document.querySelector('.filter');
    filter.classList.add('filter_open');
    this.props.infowindow.close();
  }

  //close filter
  hideFilter() {
    const filter = document.querySelector('.filter');
    filter.classList.remove('filter_open');

    this.setState({
      query: '',
      markers: this.state.currentMarkers
    });

    this.state.currentMarkers.forEach((marker) => marker.setVisible(true));
  }

  //filter
  markerFilter(e) {
    const filteredMarkers = [];
    const markers = this.state.currentMarkers;
    const query = e.target.value.toLowerCase();

    this.setState({
      query: query
    });

    if (query) {
      this.props.infowindow.close();
      markers.forEach((marker) => {
        if (marker.title.toLowerCase().indexOf(query) > -1) {
          marker.setVisible(true);
          filteredMarkers.push(marker);
        } else {
          marker.setVisible(false);
        }
      });

      filteredMarkers.sort(this.sortName);

      this.setState({
        markers: filteredMarkers
      });
    } else {
      this.setState({
        markers: this.state.currentMarkers
      });

      markers.forEach((marker) => marker.setVisible(true));
    }
  }

  openInfoWindow = (e) => {
    console.log(e);
    this.state.markers.forEach((marker) => {
      if (e.name === marker.name) {
        if (checkGetData === true) {
          this.state.infowindow.setContent(
            '<div class="info-wrap">'+
            '<h2 class="info-name">'+marker.name+'</h2><br>'+
            '<p class="info-position">Latitude: '+marker.lat+'</p>'+
            '<p class="info-position">Longitude: '+marker.lng+'</p><br>'+
            '<p class="info-address">Address: '+marker.address+'</p><br>'+
            '<img class="info-foursquare" src='+foursquareLogo+' alt="Powered by Foursquare"><br>'+
            '</div>'
          );
        } else {
          this.state.infowindow.setContent(
            '<div class="error-wrap">'+
            '<p class="error-message">Foursquare failed to load!</p><br>'+
            '</div>'
          );
        }

        this.state.infowindow.open(this.props.map, e);

        if (e.getAnimation() !== null) {
          e.setAnimation(null);
        } else {
          e.setAnimation(window.google.maps.Animation.BOUNCE);
          setTimeout(() => {
            e.setAnimation(null);
          }, 2000);
        }
      }
    });
  }

  // renders the filter, markers list and header
  render() {
    const { query, markers } = this.state;
    const { showFilter, hideFilter, markerFilter, openInfoWindow } = this;

    return (
      <aside className='wrap-filter'>
        <div
          onClick={ showFilter }
          onKeyPress={ showFilter }
          className='bFilter'
          role='button'
          arial-label="open filter"
          tabIndex="0"
          title='Open filter'>
          Open Filter
        </div>
        <h1 className='app-title'>ATL Neigborhood</h1>

        <div id='filter' className='filter'>
          <div
            onClick={ hideFilter }
            onKeyPress={ hideFilter }
            className='bFilter'
            role='button'
            aria-label="close filter"
            tabIndex="0"
            title='Close filter'>
            Close Filter
          </div>

          <input
            onChange={ markerFilter }
            className='filter-input'
            type='text'
            role='form'
            aria-labeled='filter input'
            tabIndex="0"
            placeholder='Filter ...'
            value={ query }
          />
          <ul className='filter-list'>
            {Object.keys(markers).map(i => (
              <li className='filter-item' key={ i }>
                <p
                  onClick={ () => openInfoWindow(markers[i]) }
                  onKeyPress={ () => openInfoWindow(markers[i]) }
                  onMouseOver={ () => markers[i].setIcon(markerSelected) }
                  onMouseOut={ () => markers[i].setIcon(markerUnSelected) }
                  onFocus={ () => markers[i].setIcon(markerSelected) }
                  onBlur={ () => markers[i].setIcon(markerUnSelected) }
                  className='filter-item-action'
                  role='button'
                  aria-labelledby='items to filter'
                  tabIndex="0">
                  { markers[i].name }
                </p>
              </li>
            ))}
          </ul>
        </div>
      </aside>
    );
  }
}

export default Filter;
