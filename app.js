// ************************************************************************************************************************************************
// ************************************************************************************************************************************************
// *************************************************                                    ***********************************************************
// *************************************************       VARIABLE SCATTERPLOT         ***********************************************************
// *************************************************                                    ***********************************************************
// ************************************************************************************************************************************************
// ************************************************************************************************************************************************



const buildScatterPlot = async () => {
   // here below this line is the code for Neil
   const datas = await (await fetch("/top10")).json();

   var hurricanes = datas.map(hurricane => hurricane.name_year)

   let names_years = ['Andrew_1992'];
   let name_year;
   let hurdata = {};
   let cost = 0;
   let windSpeeds = [];
   let fatalities = [];
   let hurData2 = []
   var colors = ['rgba(0, 0, 0,.5)', 'rgba(75, 93, 156,.5)', 'rgba(255, 0, 0,.5)', 'rgba(0, 161, 0,.5)', 'rgba(0, 0, 255,.5)',
      'rgb(0, 165, 255,.5)', 'rgba(0, 255, 0,.5)', 'rgba(238, 130, 238,.5)', 'rgba(255, 165, 0,.5)', 'rgba(102, 102, 102,.5)'
   ]

   datas.forEach(entry => {
      name_year = entry.name_year
      if (names_years.indexOf(name_year) > -1) {
         cost = entry.damage_usd
         windSpeeds.push(parseInt(entry.max_wind));
         fatalities.push(entry.fatalities);
         hurdata['y'] = d3.max(windSpeeds);
         hurdata['f'] = d3.max(fatalities);
         hurdata['x'] = cost;
         hurdata['r'] = cost / 2;

      } else {
         hurData2.push(hurdata)
         cost = 0;
         windSpeeds = []
         hurdata = {}
         new_name_year = `${entry.name}_${entry.year}`;
         names_years.push(new_name_year);
      }

   })
   hurData2.push(hurdata)
   
   var datasetz = []
   var dataset = {}
   for (var i = 0; i < names_years.length; i++) {
      dataset = []
      dataset['label'] = names_years[i]
      dataset['backgroundColor'] = colors[i]
      dataset['borderColor'] = colors[i]
      dataset['data'] = [hurData2[i]]
      datasetz.push(dataset)
   }

   var ctx = document.getElementById('myChart').getContext('2d');
   var scatterChart = new Chart(ctx, {
      type: 'bubble',
      data: {
         labels: names_years,
         datasets: datasetz
      },
      options: {
         scales: {
            xAxes: [{
               scaleLabel: {
                  display: true,
                  labelString: 'Damage Cost in USD (Billions)'
               }
            }],
            yAxes: [{
               scaleLabel: {
                  display: true,
                  labelString: 'Maximum WindSpeed (MPH)'
               }
            }]
         },
         tooltips: {
            callbacks: {
               label: function (tooltipItem, data) {
                  var label = data.labels[tooltipItem.index];
                  return tooltipItem.xLabel + ' Billion USD, ' + tooltipItem.yLabel + ' MPH';
               }
            }
         }
      }
   });
}

buildScatterPlot();

// ************************************************************************************************************************************************
// ************************************************************************************************************************************************
// *************************************************                                ***************************************************************
// *************************************************        GEOMAPPING PLOT         ***************************************************************
// *************************************************                                ***************************************************************
// ************************************************************************************************************************************************
// ************************************************************************************************************************************************

const buildGeomap = async () => {
   const data = await (await fetch("/top10")).json();
   
   // Define streetmap and darkmap layers
   const streetmap = L.tileLayer("https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}", {
      attribution: "© <a href='https://www.mapbox.com/about/maps/'>Mapbox</a> © <a href='http://www.openstreetmap.org/copyright'>OpenStreetMap</a> <strong><a href='https://www.mapbox.com/map-feedback/' target='_blank'>Improve this map</a></strong>",
      tileSize: 512,
      maxZoom: 18,
      zoomOffset: -1,
      id: "mapbox/streets-v11",
      accessToken: API_KEY
   });

   const dark = L.tileLayer("https://api.mapbox.com/styles/v1/mapbox/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}", {
      attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
      id: "dark-v10",
      accessToken: API_KEY
   });

   const light = L.tileLayer("https://api.mapbox.com/styles/v1/mapbox/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}", {
      attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
      maxZoom: 18,
      id: "light-v10",
      accessToken: API_KEY
   });

   // Initialize all of the LayerGroups we'll be using
   const layers = {
      Hurricane_Andrew_1992: new L.LayerGroup(),
      Hurricane_Charley_2004: new L.LayerGroup(),
      Hurricane_Harvey_2017: new L.LayerGroup(),
      Hurricane_Ike_2008: new L.LayerGroup(),
      Hurricane_Irma_2017: new L.LayerGroup(),
      Hurricane_Ivan_2004: new L.LayerGroup(),
      Hurricane_Katrina_2005: new L.LayerGroup(),
      Hurricane_Rita_2005: new L.LayerGroup(),
      Hurricane_Sandy_2012: new L.LayerGroup(),
      Hurricane_Wilma_2005: new L.LayerGroup()
   };

   // Create an initial map object
   const myMap = L.map("geomap", {
      center: [30.73, -84.50],
      zoom: 4,
      layers: [
         layers.Hurricane_Andrew_1992,
         layers.Hurricane_Charley_2004,
         layers.Hurricane_Harvey_2017,
         layers.Hurricane_Ike_2008,
         layers.Hurricane_Irma_2017,
         layers.Hurricane_Ivan_2004,
         layers.Hurricane_Katrina_2005,
         layers.Hurricane_Rita_2005,
         layers.Hurricane_Sandy_2012,
         layers.Hurricane_Wilma_2005
      ]
   });

   // Create an overlays object to add to the layer control
   const overlays = {
      "Andrew_1992": layers.Hurricane_Andrew_1992,
      "Charley_2004": layers.Hurricane_Charley_2004,
      "Harvey_2017": layers.Hurricane_Harvey_2017,
      "Ike_2008": layers.Hurricane_Ike_2008,
      "Irma_2017": layers.Hurricane_Irma_2017,
      "Ivan_2004": layers.Hurricane_Ivan_2004,
      "Katrina_2005": layers.Hurricane_Katrina_2005,
      "Rita_2005": layers.Hurricane_Rita_2005,
      "Sandy_2012": layers.Hurricane_Sandy_2012,
      "Wilma_2005": layers.Hurricane_Wilma_2005
   };

   streetmap.addTo(myMap)

   // Only one base layer can be shown at a time
   const baseMaps = {
      Light: light,
      Dark: dark,
      Streetmap: streetmap
   };

   L.control.layers(baseMaps, overlays).addTo(myMap);


   // Add function to color each hurricane
   function getColor(name_year) {
      return name_year === "Andrew_1992" ? 'rgb(0, 0, 0)' :
         name_year === "Charley_2004" ? 'rgb(75, 93, 156)' :
         name_year === "Harvey_2017" ? 'rgb(255, 0, 0)' :
         name_year === "Ike_2008" ? 'rgb(0, 161, 0)' :
         name_year === "Irma_2017" ? 'rgb(0, 0, 255)' :
         name_year === "Ivan_2004" ? 'rgb(0, 165, 255)' :
         name_year === "Katrina_2005" ? 'rgb(0, 255, 0)' :
         name_year === "Rita_2005" ? 'rgb(238, 130, 238)' :
         name_year === "Sandy_2012" ? 'rgb(255, 165, 0)' :
         'rgb(102, 102, 102)';
   }

   function getWidth(wind) {
      return wind > 150 ? 5 :
         wind > 100 ? 4 :
         wind > 50 ? 3 :
         2;
   }


   let featureType;
   let latlong = [];
   let names_years = []; //[]
   let name_year;
   let object = {} // {}
   let hurdata = {};
   let windspeed = [];

   data.forEach((entry, index) => {

      name_year = entry.name_year;
      if (names_years.indexOf(name_year) > -1) {
         var name = entry.name;
         var point = [];
         hurdata = {};
         point.push(parseFloat(entry.latitude));
         point.push(parseFloat(entry.longitude));
         latlong.push(point);
         windspeed.push(parseInt(entry.max_wind))
         hurdata['Coordinates'] = latlong;
         hurdata['WindSpeed'] = windspeed;
         hurdata['MaxWind'] = d3.max(windspeed)

         object[name_year] = hurdata;
         if (name_year === "Andrew_1992") {
            featureType = "Hurricane_Andrew_1992";
         } else if (name_year === "Charley_2004") {
            featureType = "Hurricane_Charley_2004";
         } else if (name_year === "Harvey_2017") {
            featureType = "Hurricane_Harvey_2017";
         } else if (name_year === "Ike_2008") {
            featureType = "Hurricane_Ike_2008";
         } else if (name_year === "Irma_2017") {
            featureType = "Hurricane_Irma_2017";
         } else if (name_year === "Ivan_2004") {
            featureType = "Hurricane_Ivan_2004";
         } else if (name_year === "Katrina_2005") {
            featureType = "Hurricane_Katrina_2005";
         } else if (name_year === "Rita_2005") {
            featureType = "Hurricane_Rita_2005";
         } else if (name_year === "Sandy_2012") {
            featureType = "Hurricane_Sandy_2012";
         } else {
            featureType = "Hurricane_Wilma_2005";
         }

         // extract WindSpeed parameter for getWidth function


         const newFeature = L.polyline(hurdata.Coordinates, {
            color: getColor(name_year),
            weight: 4 
         });

         // Add features to the layers according to their types
         newFeature.addTo(layers[featureType]);

         newFeature.bindPopup(`<h5>${entry.name} ${entry.year}</h5><hr>
         <p>Max. Wind: ${hurdata['MaxWind']} MPH </p>
         <p>Air Pressure: ${entry.air_pressure} mb</p>
         <p>Cost: $${entry.damage_usd} Billion</p>`, {
               maxWidth: 560
            }) //
            .addTo(myMap)


      } else {
         
         latlong = []
         windspeed = []
         hurdata = {}
         new_name_year = entry.name_year;
         names_years.push(new_name_year);

         var point = [];
         hurdata = {};
         point.push(parseFloat(entry.latitude));
         point.push(parseFloat(entry.longitude));
         latlong.push(point);
         windspeed.push(parseInt(entry.max_wind))
         hurdata['Coordinates'] = latlong;
         hurdata['WindSpeed'] = windspeed;
         hurdata['MaxWind'] = d3.max(windspeed)
         object[name_year] = hurdata;
         
         
         

         // assign feature type
         if (name_year === "Andrew_1992") {
            featureType = "Hurricane_Andrew_1992";
         } else if (name_year === "Charley_2004") {
            featureType = "Hurricane_Charley_2004";
         } else if (name_year === "Harvey_2017") {
            featureType = "Hurricane_Harvey_2017";
         } else if (name_year === "Ike_2008") {
            featureType = "Hurricane_Ike_2008";
         } else if (name_year === "Irma_2017") {
            featureType = "Hurricane_Irma_2017";
         } else if (name_year === "Ivan_2004") {
            featureType = "Hurricane_Ivan_2004";
         } else if (name_year === "Katrina_2005") {
            featureType = "Hurricane_Katrina_2005";
         } else if (name_year === "Rita_2005") {
            featureType = "Hurricane_Rita_2005";
         } else if (name_year === "Sandy_2012") {
            featureType = "Hurricane_Sandy_2012";
         } else {
            featureType = "Hurricane_Wilma_2005";
         }
  
         const newFeature = L.polyline(hurdata.Coordinates, {
            color: getColor(name_year),
            weight: 4
         });

         // Add features to the layers according to their types
         newFeature.addTo(layers[featureType]);
         newFeature.bindPopup(`<h3>${entry.name} ${entry.year}</h3><hr>
            <h4>Wind: ${entry.max_wind}</h4>
            <h4>Air pressure: ${entry.air_pressure}</h4>
            <p>Cost: ${entry.damage_usd}</p>`, {
               maxWidth: 560
            }) //
            .addTo(myMap)
         }
   })

   // Create a legend in the bottom left corner for color pallet of EQ significances
   var legend = L.control({
      position: "bottomleft"
   });
   legend.onAdd = function (myMap) {
      var div = L.DomUtil.create('div', 'legend');
      var labels = ["Andrew, 1992", "Charley, 2004", "Harvey, 2017", "Ike, 2008", "Irma, 2017", "Ivan, 2004", "Katrina, 2005", "Rita, 2005", "Sandy, 2012", "Wilma, 2005"];
      var grades = ["Andrew_1992", "Charley_2004", "Harvey_2017", "Ike_2008", "Irma_2017", "Ivan_2004", "Katrina_2005", "Rita_2005", "Sandy_2012", "Wilma_2005"];
      div.innerHTML = '<div class="legend-title">Hurricanes</br><hr></div>';
      for (var i = 0; i < grades.length; i++) {
         div.innerHTML += "<i style='background:" + getColor(grades[i]) +
            "'>&nbsp;&nbsp;</i>" + labels[i] + '<br/>';
      }
      return div;
   }
   // Add legend to map
   legend.addTo(myMap);

}

buildGeomap();


// ************************************************************************************************************************************************
// ************************************************************************************************************************************************
// *************************************************                                ***************************************************************
// *************************************************        CHOROPLETH MAP          ***************************************************************
// *************************************************                                ***************************************************************
// ************************************************************************************************************************************************
// ************************************************************************************************************************************************



const buildStateCost = async () => {
   const data = await (await fetch("/cost_by_state")).json();

   // here below this line is the code for Amy
   var states = data.map(entry => entry.name)
   var costs = data.map(entry => parseInt(entry.total_damage))
   
   var chartData = [{
      type: 'choropleth',
      locationmode: 'USA-states',
      locations: states,
      z: costs,
      text: states,
      zmin: 0,
      zmax: 50000,
      colorscale: [
         [0, '#e1e7e7'],
         [0.2, '#9a9f9f'],
         [0.4, '#848888'],
         [0.6, '#6f7171'],
         [0.8, '255, 128, 0'],
         [1, '255, 0, 0']
      ],
      colorbar: {
         title: 'Millions USD',
         thickness: 20
      },
      marker: {
         line: {
            color: 'rgb(255,255,255)',
            width: 2
         }
      }
   }];


   var layout = {
      title: 'Cumulative Hurricane Damages',
      geo: {
         scope: 'usa',
         showlakes: true,
         lakecolor: 'rgb(255,255,255)'
      }
   };

   Plotly.newPlot("costmap", chartData, layout, {
      showLink: false
   });
}


buildStateCost();

// ************************************************************************************************************************************************
// ************************************************************************************************************************************************
// *************************************************                                ***************************************************************
// *************************************************          fatalities            ***************************************************************
// *************************************************                                ***************************************************************
// ************************************************************************************************************************************************
// ************************************************************************************************************************************************

const buildFatalPlot = async () => {
   const data = await (await fetch("/fatver2")).json();

   var names = data.map(entry => entry.name);
   var years = data.map(entry => entry.year);
   var name_years = data.map(entry => `${entry.name}_${entry.year}`);
   var deaths = data.map(entry => entry.deaths);
   
     const title = ` Hurricanes with largest fatalities`;
     const trace = {
       x: name_years,
       y: deaths,
       type: 'bar',
      //  color: red, 
      //  orientation: 'h',
       title: title,
      //  text: name_years,
       marker: {
         color: 'rgb(15,52,96)'
         }

     };
     var datatrace = [trace];
     var layout = {
       title: {
         text: title,
         font: {
           size: 18
         },
       }, 
       font: {
         size: 14,
       },
       xaxis: { title: "Hurricanes" },
       yaxis: { title: "Fatalities"},
       width: 400,
       margin: {
         l: 100,
         r: 40,
         b: 100,
         t: 100,
         pad: 10}
     };
     Plotly.newPlot("plot", datatrace, layout);
 };

 buildFatalPlot();