import express from 'express'
import cors from 'cors'
import storeDocumentRoute from './routes/storeDocumentRoutes.js'
import queryDocumentRoute from './routes/queryDocumentRoutes.js'

const app = express()

// Middleware to parse JSON request bodies
app.use(express.json())

// Configure and use CORS middleware
const corsOptions = {
  origin: 'http://localhost:5173',
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization']
}

app.use(cors(corsOptions))

app.use('/store-document', storeDocumentRoute)
app.use('/query-document', queryDocumentRoute)

app.listen('8000', () => {
  console.log('Server Running on PORT 8000')
})

export default app
