import { Component, OnInit } from '@angular/core';
import {Chart} from 'chart.js'
import * as moment from 'moment';


@Component({
  selector: 'app-description',
  templateUrl: './description.component.html',
  styleUrls: ['./description.component.css']
})
export class DescriptionComponent implements OnInit {

  public row_descripteur: any = "";

  constructor() { }

  ngOnInit(): void {
      this.row_descripteur = localStorage.getItem("data");
      this.row_descripteur = JSON.parse(this.row_descripteur);
    //if (this.row_descripteur.stationnarity==true){
     // this.row_descripteur_NST=this.row_descripteur.min , this.row_descripteur.max ;
     console.log(this.row_descripteur)
  
  
var dataSample1 = JSON.parse(this.row_descripteur.dataset);   
var dataSample2 = JSON.parse(this.row_descripteur.anomalies); 
var dataSample3 = JSON.parse(this.row_descripteur.trend);
var dataSample4 = JSON.parse(this.row_descripteur.seasonal);
var dataSample5 = JSON.parse(this.row_descripteur.redsi);

var ds5 = {
  label: "Amount",
  borderColor: "rgb(51, 161, 255)",
  fill:false,
  pointRadius:0,
  borderWidth:2,
  data: dataSample5
  };

var ds4 = {
  label: "Amount",
  borderColor: "rgb(51, 161, 255)",
  fill:false,
  pointRadius:0,
  borderWidth:2,
  data: dataSample4
  };

var ds3 = {
  label: "Amount",
  borderColor: "rgb(51, 161, 255)",
  fill:false,
  pointRadius:0,
  borderWidth:2,
  data: dataSample3
  };
  
var ds1 = {
label: "AMOUNT",
borderColor: "rgb(51, 161, 255)",
fill:false,
pointRadius:0,
borderWidth:1.5,
data: dataSample1
};

var anomalies = {
  label: "Anomalies",
  borderColor: "red",
  fill:false,
  pointRadius:2,
  borderWidth:2.5,
  type :'scatter',
  data: dataSample2
  };



var chart = new Chart("DChart", {

type: 'line',
data: {datasets: [ds1]},
options: {
    scales: {
        xAxes: [{
            type: 'time',
            time: {  unit: 'day',
                     displayFormats: {
                     day: 'DD/MM/YYYY'
                },
            parser: "DD/MM/YYYY",
            }
        }]
    }
}
});



var chart = new Chart("Trend_Chart", {

  type: 'line',
  data: {datasets: [ds3]},
  options: {
    scales: {
        xAxes: [{
            type: 'time',
            time: {  unit: 'day',
                     displayFormats: {
                     day: 'DD/MM/YYYY'
                },
            parser: "DD/MM/YYYY",
            }
        }]
    }
}
 
  });

  var chart = new Chart("seasonal_Chart", {

    type: 'line',
    data: {datasets: [ds4]},
    options: {
      scales: {
          xAxes: [{
              type: 'time',
              time: {  unit: 'day',
                       displayFormats: {
                       day: 'DD/MM/YYYY'
                  },
              parser: "DD/MM/YYYY",
              max:"01/06/2017"
              }
          }]
      }
  }

 });

 
var chart = new Chart("Resid_Chart", {

  type: 'line',
  data: {datasets: [ds5]},
  options: {
      scales: {
          xAxes: [{
              type: 'time',
              time: {  unit: 'day',
                       displayFormats: {
                       day: 'DD/MM/YYYY'
                  },
              parser: "DD/MM/YYYY",
              }
          }]
      }
  }
  });

var chart = new Chart("anomliesChart", {
  type: 'line',
  data: {datasets: [ds1,anomalies]},
  options: {
      scales: {
          xAxes: [{
              type: 'time',
              time: {  unit: 'day',
                       displayFormats: {
                       day: 'DD/MM/YYYY'
                  },
              parser: "DD/MM/YYYY",
              }
          }]
      }
  }
  });
}

}
