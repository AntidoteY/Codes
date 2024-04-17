# shinyStock####
library(shiny)
library(quantmod)

ui <- fluidPage(
  titlePanel("Stock Tabsets"),
  sidebarLayout(
    sidebarPanel(
      textInput('stockname', label = 'Enter stock symbol', value = 'KO'),
      numericInput('lag', label = 'Lag for returns', 
                   value = 1, 
                   min = 1, 
                   max = 365),
      numericInput('confLevel', 
                   label = 'Confidence Level',
                   value = 0.95, min = 0.9, max = 0.99),
      numericInput('window', label = 'size',
                   value = 500, min = 1, max = 1000),
      selectInput('metric',
                  label = 'Select metric',
                  choices = c('Min', 'Max', 'Standard Deviation', 'Variance')),
      actionButton('action', 'Action')
    ),
    mainPanel(
      tabsetPanel(type = 'tabs',
                  tabPanel('Plot', plotOutput('returnPlot')),
                  tabPanel('Metric plot', plotOutput('metricPlot'))
      )
    )
  )
)

server <- function(input, output) {
  data= eventReactive(input$action, {
    req(input$stockname)
    getSymbols(Symbols = input$stockname, auto.assign = F)
  })
  
  daily_returns= reactive({
    req(data())
    dailyReturn(data())
  })
  
  output$returnPlot= renderPlot({
    returns_data= daily_returns()
    lag_returns= lag(returns_data, k = -input$lag)
    confLevel= input$confLevel
    VaR= quantile(lag_returns, 1 - confLevel, names = F, na.rm = T)
    ES= mean(lag_returns[lag_returns < VaR], na.rm = T)
    
    hist(lag_returns, main = 'Returns', 
         xlab = 'Returns', breaks = 30, col = 'blue')
    abline(v = VaR, col = 'red', lwd = 2, lty = 2)
    abline(v = ES, col = 'orange', lwd = 2, lty = 2)
    legend('topright', c(paste('VaR', round(VaR, 4)), 
                         paste('ES', round(ES, 4))), 
           col = c('red', 'orange'), lty = 2, lwd = 2)
  })
  
  output$metricPlot= renderPlot({
    req(daily_returns())
    rollings= switch(input$metric,
                     'Min' = runMin(daily_returns(), n = input$window),
                     'Max' = runMax(daily_returns(), n = input$window),
                     'Standard Deviation' = runSD(daily_returns(),
                                                  n = input$window),
                     'Variance' = (runSD(daily_returns(),n = input$window))^2
    )
    if (!is.null(rollings)) {
      plot(rollings,
           main = paste('Rolling', input$metric), 
           ylab = paste(input$metric, 'Value'), 
           xlab = 'Time')
    }
  })
}

shinyApp(ui = ui, server = server)



