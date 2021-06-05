const express = require('express');
const cors = require('cors');


class Server{
    constructor(){
        this.app = express();
        this.port = process.env.PORT;
        this.ai_path_model = '/api/ai-model';
        //Middleware
        this.middlewares();
        //Routes of my app
        this.routes();
    }

    middlewares(){
        //CORS
        this.app.use( cors() );
        //Read and parse body 
        this.app.use( express.json() );
        //Public directory
        this.app.use( express.static('public') );
    }

    routes(){
        this.app.use(this.ai_path_model, require('../routes/model_routes'));
    }

    listen(){
        this.app.listen(this.port, () => {
            console.log('Server running on port ', this.port);
        });
    }
}


module.exports = Server;