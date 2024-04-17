library(shiny)
library(car) 

ui <- fluidPage(
  titlePanel('Regression Analysis'),
  sidebarLayout(
    sidebarPanel(
      fileInput('file', 'Upload CSV File'),
      textInput('formula', 'Regression formula', value = 'y ~ variables'),
      actionButton('action', 'Run Regression')
    ),
    mainPanel(
      tableOutput('table'),
      tabsetPanel(type = 'tabs',
                  tabPanel('Data', tableOutput('datatable')),
                  tabPanel('AVPlots', plotOutput('avplots')),
                  tabPanel('VIF', verbatimTextOutput('vif')),
                  tabPanel('Influence Plots', plotOutput('inf_plots'))
      )
    )
  )
)

server <- function(input, output, session) {
  uploaded_data= reactive({
    req(input$file)
    read.csv(input$file$datapath)
  })
  
  output$datatable= renderTable({
    uploaded_data()
    
  })
  
  reg_model= reactive({
    req(input$action)
    isolate({
      formula= as.formula(input$formula)
      data= uploaded_data()
      lm(formula, data = data)
    })
  })
  
  output$avplots= renderPlot({
    req(reg_model())
    avPlots(reg_model())
  })
  
  output$vif= renderPrint({
    req(reg_model())
    vif(reg_model())
  })
  
  output$inf_plots= renderPlot({
    req(reg_model())
    influencePlot(reg_model(), id.method = 'identify', main = 'Influence Plot')
  })
}

shinyApp(ui, server)
