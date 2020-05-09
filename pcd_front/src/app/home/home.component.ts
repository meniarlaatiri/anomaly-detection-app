import { Component, OnInit } from '@angular/core';
import  {FormGroup, FormControl, FormBuilder, Validators} from '@angular/forms'
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';


@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {

  myForm: FormGroup;
  public row_descripteur : any = "";


  constructor( private http:HttpClient ,private fb: FormBuilder,  private router:Router) {
  
    let formControls = {
      format_date : new FormControl('',[Validators.required]),
    }
  
    this.myForm=this.fb.group(formControls);

  }


  ngOnInit(): void {
  }
  selectedFile:File = null ;
  onFileSelected(event){
    this.selectedFile =<File>event.target.files[0];
  }


  savefile(){
    const fd=new FormData();
    let data = this.myForm.value;
    console.log(data)
    fd.append('file',this.selectedFile, this.selectedFile.name);
    fd.append('data', JSON.stringify(data));
    this.http.post("http://127.0.0.1:5000/upload",fd).subscribe(res =>{
      this.row_descripteur= res ;
      localStorage.setItem("data",this.row_descripteur);
      this.router.navigate(['/description']);
   },
    error =>{
     console.log(error);
    }) 
  }

}
