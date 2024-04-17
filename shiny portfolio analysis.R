library(shiny)
library(quantmod)

ui <- fluidPage(
  titlePanel('Portfolio Analysis'),
  sidebarLayout(
    sidebarPanel(
      textInput('stock1', label = 'Stock1', value = 'AAPL'),
      numericInput('weight1', label = 'Weight1', value = 1/3, min = 0, max = 1),
      textInput('stock2', label = 'Stock2', value = 'GOOG'),
      numericInput('weight2', label = 'Weight2', value = 1/3, min = 0, max = 1),
      textInput('stock3', label = 'Stock3', value = 'MSFT'),
      numericInput('weight3', label = 'Weight3', value = 1/3, min = 0, max = 1),
      dateRangeInput('datename', 'Date Range', 
                     start = '2017-01-01',
                     end = '2018-01-01'),
      actionButton('action', 'Run Analysis')
    ),
    mainPanel(
      tabsetPanel(type = 'tabs',
                  tabPanel('Portfolio returns',
                           plotOutput('portfolio_returns')),
                  tabPanel('Portfolio statistics', 
                           verbatimTextOutput('portfolio_stats'))
      )
    )
  )
)
PortfolioVol <- function(S, w = NULL) {
  if (is.null(w)) {
    n = NCOL(S)
    w = rep(1 / n, n)
  }
  w = matrix(w, length(w), 1)  
  out = sqrt(t(w) %*% S %*% w)
  return(out)
  
}

server <- function(input, output, session) {
  weights= reactive({
    input_weights= c(input$weight1, input$weight2, input$weight3)
    normalized_weights= input_weights / sum(input_weights)
    return(normalized_weights)
  })
  
  portfolio_data= eventReactive(input$action, {
    req(input$stock1, input$stock2, input$stock3)
    
    stock1= getSymbols(input$stock1, 
                       auto.assign = F, 
                       from = input$datename[1], 
                       to = input$datename[2])
    stock2= getSymbols(input$stock2, 
                       auto.assign = F, 
                       from = input$datename[1], 
                       to = input$datename[2])
    stock3= getSymbols(input$stock3, 
                       auto.assign = F,
                       from = input$datename[1], 
                       to = input$datename[2])
    
    r1= periodReturn(stock1[, 6], period = "daily")
    r2= periodReturn(stock2[, 6], period = "daily")
    r3= periodReturn(stock3[, 6], period = "daily")
    
    R= merge.xts(r1, r2, r3)
    R= na.omit(R)
    df_R= as.matrix(R)
    
    current_weights= weights()  
    S= cov(df_R)
    vol= PortfolioVol(S, current_weights)
    avg_daily_returns= colMeans(df_R)
    portfolio_return= sum(current_weights * avg_daily_returns)
    sharpe_ratio= portfolio_return / vol
    
    
    list(returns = df_R, 
         volatility = vol, 
         avg_daily_returns = avg_daily_returns, 
         portfolio_return = portfolio_return, 
         sharpe_ratio = sharpe_ratio)
  })
  
  output$portfolio_returns= renderPlot({
    data= portfolio_data()
    req(data)
    
    current_weights= weights()  
    weighted_returns= data$returns %*% matrix(current_weights, ncol = 1)
    hist(weighted_returns, breaks = 30, 
         main = 'Portfolio Returns', 
         xlab = 'Returns', 
         col = 'black')
  })
  
  output$portfolio_stats= renderText({
    data= portfolio_data()
    req(data)
    
    stats= paste(
      'Average Daily Returns:\n', toString(data$avg_daily_returns),
      '\nPortfolio Volatility:\n', format(data$volatility, digits = 6),
      '\nPortfolio Return:\n', format(data$portfolio_return, digits = 6),
      '\nSharpe Ratio:\n', format(data$sharpe_ratio, digits = 6)
    )
  })
}

shinyApp(ui, server)



