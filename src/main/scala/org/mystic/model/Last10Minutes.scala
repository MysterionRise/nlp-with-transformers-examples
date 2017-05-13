package org.mystic.model

import akka.actor.Cancellable

case class Last10Minutes()

case class Last1Minute()

case class Cancel(cancel: Cancellable)

case class AskFor1MData()

case class AskFor10MData()

case class Data(data: List[Option[CandleDeal]])


