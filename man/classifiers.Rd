\name{adaptms_classifiers}
\alias{AdaptmsClassifierDF}
\alias{AdaptmsClassifierFolder}
\alias{AdaptmsMulticlassClassifier}
\alias{BaselineClassifier}
\alias{NonRefitClassifier}
\alias{NonValClassifier}
\alias{XGBRFClassifierDF}
\alias{XGBRFClassifierFolder}
\alias{install_adaptms_dependencies}
\title{ADAPT-MS Classifier Generators and Dependency Installer}
\description{
Reference class generators for ADAPT-MS classifier workflows and a helper
function to install required runtime packages.
}
\usage{
AdaptmsClassifierDF
AdaptmsClassifierFolder
AdaptmsMulticlassClassifier
BaselineClassifier
NonRefitClassifier
NonValClassifier
XGBRFClassifierDF
XGBRFClassifierFolder

install_adaptms_dependencies(
  include_optional = TRUE,
  repos = "https://cloud.r-project.org"
)
}
\arguments{
  \item{include_optional}{Logical. When \code{TRUE}, also install optional
  notebook and data IO packages.}
  \item{repos}{CRAN-style repository URL passed to
  \code{utils::install.packages()}.}
}
\details{
Each classifier object is a \code{methods::setRefClass} generator.
Instantiate with \code{$new(...)} and call the class methods documented in
the source code and project README.
}
\value{
\code{install_adaptms_dependencies()} returns (invisibly) the package names
processed for installation.
}
