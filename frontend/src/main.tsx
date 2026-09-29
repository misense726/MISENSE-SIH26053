import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import StaticDemo from './StaticDemo.tsx'

const Root = import.meta.env.MODE === 'static' ? StaticDemo : App

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)
