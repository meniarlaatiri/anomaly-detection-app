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
var ds1 = {
label: "AMOUNT",
borderColor: "rgb(51, 161, 255)",
fill:false,
pointRadius:0,
borderWidth:1.5,
data: dataSample1
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
}

}
