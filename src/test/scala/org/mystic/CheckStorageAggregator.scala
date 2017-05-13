package org.mystic

import java.time.Instant
import java.time.temporal.ChronoUnit

import akka.actor.ActorSystem
import akka.pattern.ask
import akka.testkit.{DefaultTimeout, ImplicitSender, TestActorRef, TestKit}
import org.mystic.actors.StorageActor
import org.mystic.model._
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}


class CheckStorageAggregator extends TestKit(ActorSystem("CheckAggregation"))
  with DefaultTimeout with ImplicitSender
  with WordSpecLike with Matchers with BeforeAndAfterAll {

  val actorRef = TestActorRef(new StorageActor)

  private val tradeTime = Instant.now().minus(1, ChronoUnit.MINUTES)
  private val time = tradeTime.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  actorRef ? Deal(tradeTime.toEpochMilli, "A", 100.0d, 10)
  actorRef ? Deal(tradeTime.toEpochMilli, "A", 300.0d, 10)
  actorRef ? Deal(tradeTime.toEpochMilli, "A", 200.0d, 10)

  val future = actorRef ? AskFor1MData
  val res = future.value.get.get.asInstanceOf[Data1Minutes]
  res.data should have length 1

  res.data should contain (Some(CandleDeal("A", time , 100.0d, 300.0d, 100.0d, 200.0d, 30)))


}
