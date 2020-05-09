import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';


@Component({
  selector: 'app-description',
  templateUrl: './description.component.html',
  styleUrls: ['./description.component.css']
})
export class DescriptionComponent implements OnInit {

  public row_descripteur : any = "";

  constructor() { }

  ngOnInit(): void {
    this.row_descripteur = localStorage.getItem("data");

    this.row_descripteur = JSON.parse(this.row_descripteur);
  }

}
