# useful functions to parse LMER like formula syntax
FormulaToChar <- function(formula.str) {
  if (is.character(formula.str)) {
    return(formula.str)
  }
  r <- as.character(formula.str)
  r <- c(r[2], r[1], r[-(1:2)])
  return(paste(r, collapse = " "))
}


ParseArgList <- function(str, start.at.leftparen = TRUE, delim = ",",
                         skip.whitespace = TRUE,
                         left.paren = "(", right.paren = ")") {
  # The first character must be the open parentheses "("
  args <- ""
  seps <- c()
  end.posn <- -1

  start.idx <- ifelse(start.at.leftparen, 2, 1)
  stack.init.size <- ifelse(start.at.leftparen, 1, 0)
  stack <- stack.idx <- c()
  if (start.at.leftparen) {
    stack <- substr(str, 1, 1)
    stack.idx <- which(left.paren == stack)
  }

  in.quote <- FALSE

  for (char.idx in start.idx:nchar(str)) {
    cur.char0 <- substr(str, char.idx, char.idx)
    if (in.quote) {
      if (cur.char0 == "'") {
        in.quote <- FALSE
      } else {
        args[length(args)] <- paste0(args[length(args)], cur.char0)
      }
      next
    }
    if (cur.char0 == "'") {
      in.quote <- TRUE
      next
    }
    if (length(stack) > 0
        && cur.char0 == right.paren[stack.idx[length(stack.idx)]]) {
      stack <- stack[-length(stack)]
      stack.idx <- stack.idx[-length(stack.idx)]

      if (length(stack) == 0 && start.at.leftparen) {
        end.posn <- char.idx
        break
      } else {
        args[length(args)] <- paste0(args[length(args)], cur.char0)
      }

    } else if (cur.char0 %in% left.paren) {
      stack <- c(stack, cur.char0)
      stack.idx <- c(stack.idx, which(left.paren == cur.char0))
      args[length(args)] <- paste0(args[length(args)], cur.char0)

    } else if (cur.char0 %in% delim && length(stack) == stack.init.size) {
      if (length(seps) == 0) {
        seps <- c("", seps)
      }
      args <- c(args, "")
      seps <- c(seps, cur.char0)

    } else {
      if (length(stack) > stack.init.size
          || !(skip.whitespace && cur.char0 == " ")) {
        args[length(args)] <- paste0(args[length(args)], cur.char0)
      }
    }
  }

  if (length(seps) == 0) {
    seps <- c("", seps)
  }
  return(list(args = args, end.posn = end.posn, seps = seps))
}


ParseExtraArgs <- function(args, n.remove) {
  if (length(args) > n.remove) {
    arg.lists <-
        sapply(args,
               function(arg.str) {
                 ParseArgList(arg.str, start.at.leftparen = FALSE,
                              delim = "=", skip.whitespace = TRUE)
               },
               simplify = FALSE)
    arg.list <- list()
    for (k in (n.remove+1):length(arg.lists)) {
      args <- arg.lists[[k]]$args
      arg.list[[args[1]]] <- args[2]
    }
    return(arg.list)
  } else {
    return(list())
  }
}


ParseRanef <- function(str) {
  r <- ParseArgList(str, start.at.leftparen = FALSE, delim = ":")
  return(r$args)
}


ParseFunctionCall <- function(str, left.paren = c("(", "<"),
                              right.paren = c(")", ">"), ...) {
  # The first character must be the open parentheses "("
  function.name <- ""
  end.posn <- -1

  for (char.idx in 1:nchar(str)) {
    cur.char0 <- substr(str, char.idx, char.idx)
    if (cur.char0 %in% left.paren) {
      r <- ParseArgList(substr(str, char.idx, nchar(str)),
                        start.at.leftparen = TRUE,
                        left.paren = left.paren,
                        right.paren = right.paren, ...)
      end.posn <- r$end.posn + char.idx - 1
      args <- r$args
      break
    } else {
      function.name <- paste(function.name, cur.char0, sep = "")
    }
  }
  return(list(function.name = function.name,
              args = args, end.posn = end.posn))
}


Peek <- function(str, char.idx, check.str) {
  if (char.idx > length(str)) {
    return(FALSE)
  }
  stop.idx <- min(nchar(str), char.idx + nchar(check.str) - 1)
  str0 <- substr(str, char.idx, stop.idx)
  return(str0 == check.str)
}


Trim <- function(str) {
  gsub("^\\s+|\\s+$", "", str)
}
