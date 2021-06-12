const { response } = require("express");
const { v4: uuidv4 } = require('uuid'); // extension para generar un nombre unico de fichero
const ampq = require('amqplib/callback_api');
const path = require('path');
const fs = require('fs');


const query_model = ( req, res = response) => {
    const  start   = req.query;
    //Send message to rabbit mq so as for the python process to start infering the images
    res.json({
        status:'OK',
        start
    });


}



const upload = ( req, res = response) => {
          
   
  
      // Validar que exista un archivo
      if (!req.files || Object.keys(req.files).length === 0) { 
          return res.status(400).json({
              ok: false,
              msg: 'No hay ningún archivo'
          })
        }
      
      // Procesar la imagen
  
      const file = req.files.imagen; // extraemos la imagen
  
      const nombreCortado = file.name.split('.'); // por si tenemos un fichero.1.2.3.jpg, para extraer la ultima parte .jpg
      // Obtenemos un array con cada una de las posiciones, para separarlo se usa el punto .
  
      const extensionArchivo = nombreCortado[ nombreCortado.length -1 ]; // la ultima posicion es la extension
  
      // Validar extension
  
      const extensionesValidas = ['gz', 'nii', 'nii.gz']
      if ( !extensionesValidas.includes( extensionArchivo)) {
  
          return res.status(400).json({
              ok: false,
              msg: 'No es una extensión permitida'
          });
  
      }
  
      // Si la extensión es válida, generamos un nombre unico de imagen con el paquete uuid
      const nombreArchivo = `${ uuidv4() }.${ extensionArchivo }`;
  
      // Path para guardar la imagen
  
      const path = `../data/uploads/${ nombreArchivo }`;
  
      // Mover la imagen al path
   
  
      file.mv(path, (err) => {
          if (err) {
              console.log(err);
          return res.status(500).json({
                  ok: false,
                  msg: 'Error al mover la imagen'
              });
              
          }

          res.json({
              ok: true,
              msg: 'Archivo subido',
              nombreArchivo
          });
  
      });



      
  }



module.exports = {query_model, upload};